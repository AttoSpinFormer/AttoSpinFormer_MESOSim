
#!/usr/bin/env python3
"""
###############################################################################
# Module:        MESODSTop.py
# Description:   Top-level module for MESO In-Memory Computing (IMC) operations.
#
# Synopsis:      This module handles pre-processing for IMC architectures. It 
#                accepts quantized input tensors, manages data partitioning, 
#                and applies padding to ensure proper dimensionality for 
#                crossbar array processing.
#
# Limitation:    This module is designed exclusively for single-head, multi-batch configurations.
# Reference:     For multi-head execution capability, please utilize the dedicated 
#                module: MESOTRDPTop.py.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Usage and Interface:

    The primary entry point is the 'Top' function:
    
    Out = Top(Weights, Weight_Shifts, Weight_Sign, Inputs, Input_Shift, Input_Sign, Weight_Matrix, Bit_Width)

Parameters:

    Weights:         Quantized weight tensor (e.g., Fully Connected Cell (FCC) weights, or query/key/softmax components).
    Inputs:          Quantized input tensor (e.g., FCC inputs, or key/query/value components).
    XXX_Shifts:      Exponent of the largest element relative to the defined bit-width (calculated via ShiftsConv_batch utility).
    XXX_Sign:        Sign indicator of the minimum element within the input matrix.
    Weight_Matrix:   Weight matrix with floating-point elements.
    Bit_Width:       The quantization bit-width used for the tensors.

Refer to TRDP_MESO.py for comprehensive implementation examples and usage context.
"""


import math
import sys
sys.path.append('../')
import numpy as np
import cmath
import cProfile
import torch
import torch.nn.functional as F


from DotProductMESO.Process_pytorch_MESO import VectorizedProg_torch


def Top(Weight, ShiftW, SignW, B, ShiftB, SignB, Ax, BW=8):

	# --------------------------------------------------------------------------------------------------------------------------
	# 1) Basic shape checks
	# --------------------------------------------------------------------------------------------------------------------------
	batchW,rowsW, colsW, bitsW = Weight.shape
	assert bitsW == BW, "Issues with the Weight tensor in Top module"

	batchB,rowsB, colsB, bitsB = B.shape
	assert bitsB == BW, "Issues with the Values tensor in Top module"

	assert batchW==batchB, "minibatch sizes for the inputs are different"


	# --------------------------------------------------------------------------------------------------------------------------
	# 2) Crossbar tiling parameters
	# --------------------------------------------------------------------------------------------------------------------------
	# These are the fixed crossbar dimensions used by the underlying MESO/IMC core.
	#   - `rows`: number of wordlines / rows in a crossbar
	#   - `cols`: number of bitlines / columns in a crossbar
	# The 256×64 array/tile size was selected to satisfy IR-drop constraints in a 45 nm technology node.
	# Each output element is produced by aggregating contributions from 64 MESO devices.

	cols = 64
	rows = 256

	# Current magnitude used when mapping (0/1) weight bits to +/- current.
	#A '0' maps to -I and a '1' maps to +I. 
	I=float(10E-6)

	# --------------------------------------------------------------------------------------------------------------------------
	# 3) Reorder and map weights to +/-I
	# --------------------------------------------------------------------------------------------------------------------------
	# Move bit-slices forward so Weight becomes: (batch, BW, rowsW, colsW)
	# The +/- I values are subsequently divided onto the finite sized crossbars. 

	Weight=Weight.permute(0,3,1,2)

	# Map stored bits/levels to a sign/current model expected by the MESO core.
	# Here: 0 -> -I, non-zero -> +I.
	Weight=torch.where(Weight==0,-I,I)


	# --------------------------------------------------------------------------------------------------------------------------
	# 4) Flatten B's (colsB, BW) into a single feature dimension
	# --------------------------------------------------------------------------------------------------------------------------
	# (batch, rowsB, colsB, BW) -> (batch, rowsB, colsB*BW)
	# Each row now carries all bit-slices of the original features.
	B=B.reshape(batchB,rowsB,-1)


	# --------------------------------------------------------------------------------------------------------------------------
	# 5) Tile Weight into (D1 x D2) crossbar blocks and pad to crossbar multiples - emulating crossbar programming
	# --------------------------------------------------------------------------------------------------------------------------
	# Number of crossbars needed along Weight's two spatial dimensions.

	D1 = math.ceil(rowsW / rows)
	D2 = math.ceil(colsW / cols)
    
	# Amount of zero/constant padding needed to reach exact crossbar multiples.
	padX = int(rows * D1) - rowsW #pad along Weight row dimension
	padY = int(cols * D2) - colsW #pad along Weight col dimension
    
	# F.pad uses the reverse order of dimensions; for 4D tensors we supply 8
	# values (pad_left, pad_right) for each of the last 4 dimensions.
	# We only pad (rowsW, colsW) and keep (batch, BW) unchanged
	# The entire dot-product operation gets performed over (batchW x BW x D1 x D2) crossbars. 
	# The crossbars are of size rows x cols
	padding = (0, padY, 0, padX, 0, 0, 0, 0) 
    
	# Pad with -I (the "zero" weight current in this representation) to make the 
	#new dimension of W_padded a multiple of the crossbar size. 
	W_padded = F.pad(Weight, padding, mode='constant', value=-I) 
    
	#Reshape into explicit tiles:
	# (batch, BW, rowsW_pad, colsW_pad) -> (batch, BW, D1, rows, D2, cols)
	WeightN = W_padded.reshape(batchW, BW, D1, rows, D2, cols)
    
	#Permute to group crossbars as (D1, D2) with the per-crossbar (rows, cols) last: (batch, BW, D1, D2, rows, cols)
	WeightN = WeightN.permute(0, 1, 2, 4, 3, 5)


	# --------------------------------------------------------------------------------------------------------------------------
	# 6) Tile B in blocks of 64 rows to match Weight column tiles
	# --------------------------------------------------------------------------------------------------------------------------
	# B is blocked by `cols` (64) because it is given as gate voltage to the transistors along each column of the crossbar.
	num_blocks_B = math.ceil(rowsB / cols)
	pad_rows_B = int(num_blocks_B * cols) - rowsB
    
	# Pad only the row dimension of B (second-to-last dim for a 3D tensor).
	padding_B = (0, 0, 0, pad_rows_B, 0, 0)
	B_padded = F.pad(B, padding_B, mode='constant', value=0.0)

	#Block rows into (num_blocks_B, 64) chunks: (batch, rowsB_pad, colsB*BW) -> (batch, num_blocks_B, cols, colsB*BW)
	#This is done to increase the ease of the CIM-based dot-product execution. 
	BN = B_padded.reshape(batchB, num_blocks_B, cols, colsB*BW)

	# Assertion based Sanity check: Weight column blocks must match B row blocks.
	assert D2==num_blocks_B

	# --------------------------------------------------------------------------------------------------------------------------
	# 7) Call the MESO vectorized core that performs the in-memory processing, ADC conversion and post-processing.
	# --------------------------------------------------------------------------------------------------------------------------
	Output = VectorizedProg_torch(WeightN, ShiftW, SignW, BN, ShiftB, SignB, Ax, BW)

	return Output



