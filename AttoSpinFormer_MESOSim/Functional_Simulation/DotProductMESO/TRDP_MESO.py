

#!/usr/bin/env python3
"""
###############################################################################
# Module:        TRDP_MESO.py
# Description:   Top-level module interfacing the MESO IMC pipeline with Attention Mechanisms.
#
# Synopsis:      This module manages the transition from floating-point attention 
#                layer outputs to IMC-compatible inputs. It performs the necessary 
#                quantization and binarization, calculates the required shift 
#                and sign metadata for the tensors, and orchestrates the submission 
#                of these processed inputs and metadata to the core IMC module (MESOTRDPTop.py).
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################

Usage and Interface:

    This module provides two specialized classes for the Attention mechanism's matrix multiplications:

    1. FPMatMulMESO4D (Class): Performs the **Queries x Keys_T** operation.
    2. FPMatMulMESO4D2 (Class): Performs the **Softmax x Values** operation.

    Constructor (FPMatMulMESO4D):
    constructor: FPMatMulMESO4D(Bit_Width: int)
    Method Call Inputs: Queries (torch.Tensor), Keys (torch.Tensor).

    Constructor (FPMatMulMESO4D2):
    constructor: FPMatMulMESO4D2(Bit_Width: int)
    Method Call Inputs: Softmax (torch.Tensor), Values (torch.Tensor).
    
Parameters:

    Bit_Width :         Defines the quantization bit-width applied to all processed tensors.
    Input Dimension :   Specifies the input tensor shape as Batch x Heads x Rows x Columns (B x H x R x C).

Consult TestScripts/Transformer_Simulation/scr/Attention.py for comprehensive implementation examples and operational context.
"""


import math
import sys
sys.path.append('../')
import cmath
import torch
import torch.nn as nn
import torch.nn.functional as F
import cProfile

from DotProductMESO.MESOTRDPTop import Top
from DotProductMESO.Process_TR_MESO import VectorizedProg_torch

from GeneralScripts.ShiftsConv_batch import Shifts_torch3
from GeneralScripts.ShiftsConv_batch import converge4

from torch.autograd import Function


#Queries is the Weights and Keys is the Inputs

class FPMatMulAutogradMESO(Function):
	@staticmethod
	def forward(ctx, Queries, Keys, module_instance):
		ctx.save_for_backward(Queries, Keys)
		ctx.module_instance = module_instance 

		BT= Keys.permute(0,1,3,2)
		Weight=Queries

		batchW, multi_headW,rowsW, colsW=Weight.shape
		batchBT, multi_headBT, rowsBT, colsBT=BT.shape

		assert colsW==rowsBT, "dimensional mismatch between inputs"
		assert batchW==batchBT, "batch size mismatch between inputs"
		assert multi_headW==multi_headBT, "heads mismatch between inputs"
			
		BW = module_instance.BW 

		WBin,minW,ShiftW,SignW = Shifts_torch3(Weight, BW)
		BTBin,minBT,ShiftBT,SignBT = Shifts_torch3(BT, BW)

		WBinMerge,BTBinMerge=converge4(WBin,minW,BTBin,minBT)
		Output = Top(WBinMerge, ShiftW, SignW, BTBinMerge, ShiftBT, SignBT, Weight, BW)
		return Output.to(Queries.dtype)

	@staticmethod
	def backward(ctx, grad_output):
		Queries, Keys = ctx.saved_tensors
		B, H, L_Q, D_K = Queries.shape
		L_K = Keys.shape[2] 

		#print("In autograd")
		#print(B,H,L_Q,D_K, L_K, Keys.shape[3])

		# Reshape Inputs for torch.bmm: (B, H, L, F) -> (B*H, L, F)
		queries_3d = Queries.reshape(B * H, L_Q, D_K)
		keys_3d = Keys.reshape(B * H, L_K, D_K)
		grad_output_3d = grad_output.reshape(B * H, L_Q, L_K)

		#print(grad_output.shape)
		#print(grad_output_3d.shape)
    
		# Gradient for Queries (grad_Q) = grad_output @ Keys.T
		grad_queries_3d = torch.bmm(grad_output_3d, keys_3d)
    
		# Gradient for Keys (grad_K) = grad_output.T @ Queries
		grad_keys_3d = torch.bmm(grad_output_3d.transpose(1, 2), queries_3d)

		# Reshape Gradients back to 4D
		grad_queries = grad_queries_3d.reshape(B, H, L_Q, D_K)
		grad_keys = grad_keys_3d.reshape(B, H, L_K, D_K)
    
		return grad_queries, grad_keys, None


class FPMatMulMESO4D(nn.Module):
	def __init__(self,bit_width=16):
		super().__init__()
		self.BW=bit_width

	def forward(self,Queries,Keys):
		return FPMatMulAutogradMESO.apply(Queries,Keys,self)


class FPLinearMESO(nn.Module):
	def __init__(self, in_features, out_features, bit_width=8):
		super(FPLinearMH7, self).__init__()
		self.weight = nn.Parameter(torch.randn(out_features,in_features)*2/torch.sqrt(torch.tensor(in_features)))
		self.bias = nn.Parameter(torch.randn(out_features))
		self.fixed_point_mm = FPMatMulMESO(bit_width=bit_width)

	def forward(self, input):
		output = self.fixed_point_mm(input, self.weight).T+ self.bias
		return output




class FPMatMulAutogradMESO2(Function):
	@staticmethod
	def forward(ctx, Queries, Keys, module_instance):
		ctx.save_for_backward(Queries, Keys)
		ctx.module_instance = module_instance 

		BT= Keys
		Weight=Queries

		batchW, multi_headW,rowsW, colsW=Weight.shape
		batchBT, multi_headBT, rowsBT, colsBT=BT.shape

		assert colsW==rowsBT, "dimensional mismatch between inputs"
		assert batchW==batchBT, "batch size mismatch between inputs"
		assert multi_headW==multi_headBT, "heads mismatch between inputs"
			
		BW = module_instance.BW 

		WBin,minW,ShiftW,SignW = Shifts_torch3(Weight, BW)
		BTBin,minBT,ShiftBT,SignBT = Shifts_torch3(BT, BW)

		WBinMerge,BTBinMerge=converge4(WBin,minW,BTBin,minBT)
		Output = Top(WBinMerge, ShiftW, SignW, BTBinMerge, ShiftBT, SignBT, Weight, BW)
		return Output.to(Queries.dtype)

	@staticmethod
	def backward(ctx, grad_output):
		Queries, Keys = ctx.saved_tensors
		B, H, L_Q, L_K = Queries.shape
		D_K = Keys.shape[3] 

		# Reshape Inputs for torch.bmm: (B, H, L, F) -> (B*H, L, F)
		queries_3d = Queries.reshape(B * H, L_Q, L_K)
		keys_3d = Keys.reshape(B * H, L_K, D_K)
		grad_output_3d = grad_output.reshape(B * H, L_Q, D_K)

		#print(grad_output.shape)
		#print(grad_output_3d.shape)
    
		# Gradient for Queries (grad_Q) = grad_output @ Keys.T
		grad_queries_3d = torch.bmm(grad_output_3d, keys_3d.transpose(1,2))
    
		# Gradient for Keys (grad_K) = grad_output.T @ Queries
		grad_keys_3d = torch.bmm(queries_3d.transpose(1, 2), grad_output_3d)

		# Reshape Gradients back to 4D
		grad_queries = grad_queries_3d.reshape(B, H, L_Q, L_K)
		grad_keys = grad_keys_3d.reshape(B, H, L_K, D_K)

		return grad_queries, grad_keys, None


class FPMatMulMESO4D2(nn.Module):
	def __init__(self,bit_width=16):
		super().__init__()
		self.BW=bit_width

	def forward(self,Queries,Keys):
		return FPMatMulAutogradMESO2.apply(Queries,Keys,self)


