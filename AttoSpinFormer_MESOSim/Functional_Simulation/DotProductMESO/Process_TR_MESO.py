

#!/usr/bin/env python3
"""
###############################################################################
# Module:        Process_MESO.py
# Description:   Core PyTorch module for MESO In-Memory Computing (IMC) kernel execution.
#
# Synopsis:      This module orchestrates the IMC pipeline: accepting split
#                quantized weight and input matrices, simulating crossbar 
#                computation, and converting analog outputs (Current -> Voltage) 
#                into digital floating-point results. It incorporates the
#                required compensation term before reorganizing the outputs 
#                to their final, correct tensor dimensions.
#		 This module is designed to handle multi-batch, multi-head configurations.
#
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

Refer to MESOTRDPTop.py for upstream data handling and comprehensive implementation examples.
"""


import math
import sys
sys.path.append('../')

import torch

def VectorizedProg_torch(A1, P1, SignA, B1, P2, SignB, Ax, n=8):

	I=float(10E-6)

	batchB, headsB, rowbatchB, rowsB, colsB = B1.shape
	batchA, headsA, bitsA, rowbatchA, colsbatchA, rowsA, colsA =A1.shape

	#If you are changing colsA in MESODSTop.py, change the following variables
	ADCres=1+torch.log2(torch.tensor(colsA).float()).to(torch.int32).item() #=log2(colsA)+1
	Res=float(pow(2,ADCres))/2
	R=float(1)/(I*colsA)

	assert colsbatchA==rowbatchB, "dimensionality mismatch"
	assert batchA==batchB, "minibatch mismatch"
	assert headsA==headsB, "heads mismatch"

	device = A1.device
	dtype = A1.dtype

	bCols= colsB// bitsA

	out=torch.einsum('abtxjlm,abjmu->abtxjlu', A1, B1)

	out=out*R
	out=(out+1)*Res
	out=torch.round(out)-Res

	out=out.sum(axis=4)

	batchF, headsF, rowsF, colsF=Ax.shape

	mod=B1.sum(axis=2)
	mod=mod.sum(axis=2)
	
	out=out+mod.unsqueeze(2).unsqueeze(3).unsqueeze(4)

	out=out//2

	factor=torch.pow(2, torch.arange(bitsA, dtype=out.dtype, device=device))

	outNew=torch.sum(out * factor.reshape(1, 1,-1, 1, 1, 1), dim=2)

	outNew_reshaped=outNew.reshape(batchA, headsA, rowbatchA, rowsA, bCols, bitsA)
	
	out3=torch.sum(outNew_reshaped * factor, dim=-1)

	out3_reshaped=out3.reshape(out3.shape[0], out3.shape[1], -1, out3.shape[4])


	out4_optimized = out3_reshaped[:, :, :(rowsF+1), :]

	weighted_row_term = SignA * out4_optimized[:, :, rowsF, :bCols-1].unsqueeze(2)
	weighted_col_term = SignB * out4_optimized[:, :, :rowsF, bCols-1].unsqueeze(3)

	base_slice = out4_optimized[:, :, :rowsF, :bCols - 1]
	out4x = base_slice + weighted_row_term + weighted_col_term

	scalar_term = SignB * SignA * out4_optimized[:, :, rowsF, bCols-1]

	out4x_Final=out4x+scalar_term.unsqueeze(2).unsqueeze(3)

	return out4x_Final*pow(2,-P1-P2+2)

