
#!/usr/bin/env python3
"""
###############################################################################
# Module:        Attention.py
# Description:   Multi-head attention layer implementation for the Small Language Model (SLM) architecture.
#
# Synopsis:      This module implements the attention layer's matrix multiplication, 
#                acting as a functional switch based on the global execution mode.
#                It selectively executes either the **MESO IMC-based dot-product** 
#                kernel (when mode = 1) or the standard **CMOS GPU-based dot-product** 
#                implementation (when mode = 0) for comparative analysis.
#
# Created:       2025-11-11
# Last Modified: 2025-12-08
###############################################################################
"""


import torch
from torch import nn
import math

import sys

sys.path.append('../../../')

from DotProductMESO.TRDP_MESO import FPMatMulMESO4D
from DotProductMESO.TRDP_MESO import FPMatMulMESO4D2






class MultiHeadAttention(nn.Module):

	def __init__(self, d_model, n_head, mode, bit_width):
		super(MultiHeadAttention, self).__init__()
		self.n_head = n_head
		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)
		###
		self.attention = ScaleDotProductAttention(mode, bit_width)
		self.w_concat = nn.Linear(d_model, d_model)

	def forward(self, q, k, v, mask=None):
		q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
		q, k, v = self.split(q), self.split(k), self.split(v)
		out= self.attention(q, k, v, mask=mask)
		out = self.concat(out)
		out = self.w_concat(out)
		return out

	def split(self, tensor):
		"""
		split tensor by number of head

		:param tensor: [batch_size, length, d_model]
		:return: [batch_size, head, length, d_tensor]
		"""
		batch_size, length, d_model = tensor.size()

		d_tensor = d_model // self.n_head
		#we are doing this transposing because then we can use in-built MMM operations. The MMM is performed between the last two dimensions. view basically changes the dimensions into what we want. 
		tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
		return tensor

	def concat(self, tensor):
		"""
		inverse function of self.split(tensor : torch.Tensor)

		:param tensor: [batch_size, head, length, d_tensor]
		:return: [batch_size, length, d_model]
		"""
		batch_size, head, length, d_tensor = tensor.size()
		d_model = head * d_tensor

		#for transmission of data, we retranspose the data into the initial form. 
		#contiguous is basically a memory format. For ease, we use the same format for all the transmitted input data. 
		tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
		return tensor



class ScaleDotProductAttention(nn.Module):

	def __init__(self,mode, bit_width):
		super(ScaleDotProductAttention,self).__init__()
		self.softmax=nn.Softmax(dim=-1)
		self.mode=mode
		self.qktdp=FPMatMulMESO4D(bit_width=bit_width)
		self.smvadp=FPMatMulMESO4D2(bit_width=bit_width)


	def forward(self,q,k,v,mask=None,e=1e-12):
		batch_size,head,length,d_tensor=q.size()
		k_t=k.transpose(2,3)

		if self.mode==1:
			score= self.qktdp(q,k) / math.sqrt(d_tensor)
		else:
			score=q@k_t/math.sqrt(d_tensor)
		
		if mask is not None:
			score=score.masked_fill(mask==0, -1e9)

		score=self.softmax(score)
		if self.mode==1:
			score=self.smvadp(score,v)
		else:
			score=score@v
		return score


class FFN(nn.Module):
	def __init__(self,d_model,hidden,output,drop_prob=0.1):
		super(FFN,self).__init__()
		self.f1=nn.Linear(d_model,hidden)
		self.f2=nn.Linear(hidden,output)
		self.Relu=nn.ReLU()
		self.dropout=nn.Dropout(p=drop_prob)


	def forward(self,x):
		x=self.f1(x)
		x=self.Relu(x)
		x=self.dropout(x)
		x=self.f2(x)
		return x