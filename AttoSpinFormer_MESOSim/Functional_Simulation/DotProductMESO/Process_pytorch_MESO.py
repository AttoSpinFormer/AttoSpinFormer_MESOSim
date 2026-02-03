
#!/usr/bin/env python3
"""
###############################################################################
# Module:        Process_pytorch_MESO.py
# Description:   Core PyTorch module for MESO In-Memory Computing (IMC) kernel execution.
#
# Synopsis:      This module orchestrates the IMC pipeline: accepting split
#                quantized weight and input matrices, simulating crossbar 
#                computation, and converting analog outputs (Current -> Voltage) 
#                into digital floating-point results. It incorporates the
#                required compensation term before reorganizing the outputs 
#                to their final, correct tensor dimensions.
#
# Limitation:    This module is designed exclusively for single-head configurations.
# Reference:     For multi-head execution capability, please utilize the dedicated 
#                module: Process_TR_MESO.py.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Usage and Interface:

    The primary entry point is the 'VectorizedProg_torch' function, which operates on PyTorch tensors:
    
    Out = VectorizedProg_torch(Weights_Split, Weight_Shifts, Weight_Sign, Inputs_Split, 
                               Input_Shift, Input_Sign, Weight_Matrix, Bit_Width)

Parameters:

    Weights_Split (torch.Tensor): Quantized weight tensor, partitioned into fixed 256x64 blocks for parallel crossbar execution 
                                  (e.g., FCC weights, attention queries/keys/softmax).
    Inputs_Split (torch.Tensor):  Quantized input tensor, partitioned for effective IMC parallelization.
    XXX_Shifts (torch.Tensor):    Exponent of the largest element relative to the defined Bit_Width (calculated via ShiftsConv_batch utility).
    XXX_Sign (torch.Tensor):      Sign indicator of the minimum element within the weight matrix used for sign reconstruction.
    Weight_Matrix (torch.Tensor): The original floating-point weight matrix used for reference and compensation.
    Bit_Width (int):              The quantization bit-width applied to the input and weight tensors.

Dependencies:
    Requires torch.Tensor inputs for all relevant parameters.

Refer to MESODSTop.py for upstream data handling and comprehensive implementation examples.
"""


import math
import sys
sys.path.append('../')

import torch

def VectorizedProg_torch(A1, P1, SignA, B1, P2, SignB, Ax, n=8):

	"""
	Run the processing and post-processing cycles of the MESO/CIM vectorized dot-product (matmul) 
	pipeline on quantized/split tensors.

	This function models the IMC datapath at a high level:
	  1. Computes per-tile analog dot-products using a crossbar-style einsum.
	  2. Applies an ADC quantization model (Current -> Voltage scaling + discretization).
	  3. Recombines bit-slices using exponent/mantissa alignment metadata (P1/P2).
	  4. Adds the compensation terms required by the "shift-by-min" tensor decomposition
	    (used to make operands non-negative for CIM).
	  5. Adds compensation terms to account for the negative current of MESO for a logical '0' (check the AttoSpinFormer paper). 

	Expected tensor shapes
	---------------------
	A1: (batch, bit_width, row_tiles_A, col_tiles_A, crossbar_rows, crossbar_cols)
	B1: (batch, row_tiles_B, crossbar_cols, colsB_total)
	where col_tiles_A == row_tiles_B and crossbar_cols matches between A1/B1 for matmul.

	Args:
		A1: Shifted + Quantized + split weight tensor (bit-sliced and tiled).
		P1: Weight shift/exponent metadata (from mantissa alignment).
		SignA: Sign of the minimum/offset term for the weight decomposition (+/- 1).
		B1: Shifted + Quantized + split activation/input tensor (tiled to match A1).
		P2: Input shift/exponent metadata (from mantissa alignment).
		SignB: Sign of the minimum/offset term for the input decomposition (±1).
		Ax: Reference tensor used to recover the intended output size. 
		n:  bit_width.

	Returns:
		torch.Tensor: Reconstructed matmul result with compensation terms applied.
	"""

	# Reference MESO output current. Used to normalize the accumulated analog current sum into a
	# voltage before ADC discretization.
	I=float(10E-6)
	dtype=A1.dtype
	
	# Unpack tensor shapes used throughout the pipeline.
	# B1 is a 4-D tiled activation tensor; 
	#A1 is a 6-D bit-sliced + tiled weight tensor.
	batchB,rowbatchB, rowsB, colsB = B1.shape
	batchA,bitsA, rowbatchA, colsbatchA, rowsA, colsA =A1.shape

	
	# ADC resolution model: chooses enough bits to represent a sum of 'colsA' terms.
	# For a sum of N terms, a common bound is log2(N)+1 bits.
	ADCres=1+torch.log2(torch.tensor(colsA).float()).to(torch.int).item() #=log2(colsA)+1

	#Convert ADC resolution into a quantization scale. 'Res' is half of the full-scale
	# range for the signed ADC code used below.
	Res=math.ceil(pow(2,ADCres)/2)

	# R models the load resistance connected to the sense amplifier, which converts the
	# accumulated crossbar output current into a voltage.
	# It is derived by normalizing against the maximum possible accumulated current
	# (I * colsA), ensuring the resulting analog accumulation remains comparable
	# across different dot‑product lengths.
	R=float(1)/(I*colsA)

	#Assertion-based Sanity checks: A1/B1 must agree on the number of tiles along the reduction axis,
	#and batch dimensions must match.
	assert colsbatchA==rowbatchB, "dimensionality mismatch"
	assert batchA==batchB, "minibatch mismatch"

	device = A1.device

	# Columns are grouped by bit-slices. 'bCols' is the effective number of output column
	# groups once bit-sliced packing is accounted for.
	bCols= colsB// bitsA
	

	# Crossbar-style multiply-accumulate (vectorized):
	#   A1 carries (bitsA) bit-slices and tiling in both row/col;
	#   B1 is tiled to match the A1 column-tiles (reduction axis).
	# The einsum produces per-crossbar dot-products with explicit bit-slice dimension preserved.
	out=torch.einsum('btxjlm,bjmu->btxjlu', A1, B1)

	# Current -> voltage model: use the resistor R connected to the sense amplifier.
	out=out*R
	
	# Analog -> digital model: scale by R, then apply an ADC transfer/quantization model.
	# Convert analog voltage values to discrete digital integers.
	# Map normalized values into ADC code space. The +1 shift centers the range so that
	# subsequent flooring behaves like a mid-tread quantizer. This is similar to typical ADC behavior.
	out=(out+float(1))*Res

	# Discretize to integer ADC codes and re-center around zero - this is extra compensation.
	out=torch.round(out)-Res

	# Sum across the A1 'col-tiles' dimension (axis=3 in the einsum output) to accumulate
	# partial dot-products into full dot-products for each row-tile and output column.
	out=out.sum(axis=3).to(dtype)

	# Apply a correction term derived from the input coding.
	# This is a compensation term to account for the negative current flow in MESO devices. 
	mod=B1.sum(axis=1)
	mod=mod.sum(axis=1)
	out=out+mod.unsqueeze(1).unsqueeze(2).unsqueeze(3)


	# Rescale after correction (integer right shift). This effectively divides by 2 and is
	# also a compensation term to account for the negative MESO current flow (check the AttoSpinFormer paper)
	out=out//2
	#out=out>>1

	batchF,rowsF,colsF=Ax.shape


	# Bit-slice recombination factor. Each bit-slice is weighted by its significance and
	# the mantissa-alignment shifts (P1, P2).
	# Note: P1 and P2 are typically per-(batch,row/col tile) metadata; broadcasting relies
	# on PyTorch's implicit expansion rules.
	# P1 : Shift term for the weight tensor
	# P2: Shift term for the activations tensor.
	factor=torch.pow(2, torch.arange(bitsA, dtype=torch.int32, device=device)-(0.5*(P1+P2))+1).to(dtype)

	# Combine the bit-slice dimension (dim=1) into a single fixed-point value per tile.
	outNew=torch.sum(out * factor.reshape(1,-1, 1, 1, 1), dim=1, dtype=dtype)


	# Repack the column dimension into (bCols, bitsA) so we can apply a second-stage
	# recombination across the packed bit-slices per output column group.
	outNew_reshaped=outNew.reshape(batchA, rowbatchA, rowsA, bCols, bitsA)
	
	# Final bit-weighted accumulation across the packed bit dimension.
	out3=torch.sum(outNew_reshaped * factor, dim=-1, dtype=dtype)

	# Collapse tiled rows into a single contiguous row dimension.
	out3_reshaped=out3.reshape(out3.shape[0], -1, out3.shape[3])

	# Trim any padding introduced by upstream tiling. Upstream typically appends an extra
	# row/column that store the decomposition offset terms needed for reconstruction.
	out4_optimized = out3_reshaped[:, :(rowsF+1), :]

	# Reconstruction for shift-by-min decomposition:
	#   Upstream decomposes each operand as:
	#       X' = X - SignX * ABX,   where SignX = sign(min(X)) and ABX = |min(X)|
	#       Y' = Y - SignY * ABY,   where SignY = sign(min(Y)) and ABY = |min(Y)|
	#
	#   Expanding the original product gives:
	#       X * Y = X'Y' + (SignX*ABX)*Y' + (SignY*ABY)*X' + (SignX*SignY*ABX*ABY)
	#
	#   `out4_optimized` reserves its last row/column to carry the correction terms required to
	#   reconstruct X*Y from the shifted products: (SignX*ABX)*Y', (SignY*ABY)*X', and
	#   SignX*SignY*ABX*ABY.
	#
	#   The `converge` function performs the post shift + quantization packing needed to expose
	#   these compensation terms to the CIM/MESO datapath.
	weighted_row_term = SignA * out4_optimized[:, rowsF, :bCols-1].unsqueeze(1)
	weighted_col_term = SignB * out4_optimized[:, :rowsF, bCols-1].unsqueeze(2)

	base_slice = out4_optimized[:, :rowsF, :bCols - 1]
	out4x = base_slice + weighted_row_term + weighted_col_term

	# Scalar offset term (min(X)*min(Y)) applied uniformly across all outputs in this tile.
	scalar_term = SignB * SignA * out4_optimized[:, rowsF, bCols-1]

	# Final reconstructed result (base + row term + col term + scalar term).
	out4x_Final=out4x+scalar_term.unsqueeze(1).unsqueeze(2)

	return out4x_Final.to(dtype)

