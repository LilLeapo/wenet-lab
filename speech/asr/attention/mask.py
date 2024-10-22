import math
from typing import Optional, Tuple

import torch
from torch import nn

def get_mask_for_encoder(input_lens,padded_lens=None):
    '''

    :param input_lens: list of int
    :param padded_lens: int
    :return:
    '''
    if padded_lens is None:
        padded_lens=int(max(input_lens))
    batch_size=len(input_lens)
    seq_range=torch.arange(padded_lens)
    input_lens=torch.tensor(input_lens)
    seq_range_expand = seq_range.unsqueeze(0)\
        .expand(batch_size, padded_lens)
    seq_length_expand = input_lens.unsqueeze(-1)  # (B,1)
    # 补全代码4:padding的位置对应为False,真实数据的位置为True(可以使用比较运算符实现)
    mask =          # (B,max_len)
    mask=mask.unsqueeze(-2)
    return mask # (B,1,max_len)

def get_mask_for_decoder(padded_lens,):
    ret = torch.ones(padded_lens, padded_lens,dtype=torch.bool)
    # 补全代码5:⽣成下三⾓矩阵(调用torch中的函数)
    mask=
    return mask

if __name__=='__main__':
    print(get_mask_for_encoder([1,2,3]))
    print(get_mask_for_decoder(10))