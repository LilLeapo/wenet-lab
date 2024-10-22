import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head

        self.W_q = torch.nn.Parameter(torch.randn(n_head, n_feat, self.d_k))
        self.B_q = torch.nn.Parameter(torch.randn(n_head, 1, self.d_k))

        self.W_k = torch.nn.Parameter(torch.randn(n_head, n_feat, self.d_k))
        self.B_k = torch.nn.Parameter(torch.randn(n_head, 1, self.d_k))

        self.W_v = torch.nn.Parameter(torch.randn(n_head, n_feat, self.d_k))
        self.B_v = torch.nn.Parameter(torch.randn(n_head, 1, self.d_k))

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def step_1(self,query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        q = query.unsqueeze(1) @ self.W_q + self.B_q
        k= key.unsqueeze(1) @ self.W_k + self.B_k
        v = value.unsqueeze(1) @ self.W_v + self.B_v
        return q,k,v
    def step_2(self,q,k,mask=None):
        # 补全代码1:计算注意⼒评分(使用torch中的函数计算)
        # TO DO
        scores = 
        if mask is not None:
            scores = scores.masked_fill(mask, -float('inf'))
        return scores
    def step_3(self,scores,mask=None):
        # 补全代码2:计算注意⼒评分在最后一维上的概率分布(使用torch中的函数)
        # TO DO
        attn = 
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        return attn
    def step_4(self,attn,v):
        # 补全代码3:计算注意⼒头的输出(使用矩阵乘法)
        output_h=
        return output_h
    def step_5(self,output_h):
        n_batch=output_h.shape[0]
        output=output_h.transpose(1, 2)\
             .contiguous()\
             .view(n_batch, -1,self.h * self.d_k)
        return self.linear_out(output)
    def forward(self,query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
        q, k, v=self.step_1(query,key, value)
        scores = self.step_2(q, k, mask)
        attn=self.step_3(scores, mask)
        output_h=self.step_4(attn,v)
        outputs=self.step_5(output_h)
        return outputs

