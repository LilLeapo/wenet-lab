import torch
import math
from torch import nn
from attention import MultiHeadedAttention
from mask import get_mask_for_encoder,get_mask_for_decoder
from matplotlib import pyplot as plt
B=3
T=10
D=16
H=4

if True:
    exp_2_inputs=torch.load('exp_2/exp_2_inputs.pt')
    mha = MultiHeadedAttention(n_head=H, n_feat=D, dropout_rate=0)
    mha.load_state_dict(exp_2_inputs['mha'])
    query=exp_2_inputs['query']
    key=exp_2_inputs['key']
    value=exp_2_inputs['value']
    input_lens=exp_2_inputs['input_lens']
    padded_lens=exp_2_inputs['padded_lens']

    mask=get_mask_for_encoder(input_lens,padded_lens)

    output_all=mha(query,key,value,mask)
    output_error = mha(query, key, value)

    print(torch.dist(output_all,output_error))


    output_1=mha(query[0,:input_lens[0]].unsqueeze(0),
                 key[0,:input_lens[0]].unsqueeze(0),
                 value[0,:input_lens[0]].unsqueeze(0))
    loss_1=torch.dist(output_all[0,:input_lens[0]].unsqueeze(0),output_1)
    loss_1_ = torch.dist(output_error[0, :input_lens[0]].unsqueeze(0), output_1)
    print(loss_1, loss_1_)

    output_2=mha(query[1,:input_lens[1]].unsqueeze(0),
                 key[1,:input_lens[1]].unsqueeze(0),
                 value[1,:input_lens[1]].unsqueeze(0))
    loss_2=torch.dist(output_all[1,:input_lens[1]].unsqueeze(0),output_2)
    loss_2_ = torch.dist(output_error[1, :input_lens[1]].unsqueeze(0), output_2)
    print(loss_2, loss_2_)

    output_3=mha(query[2,:input_lens[2]].unsqueeze(0),
                 key[2,:input_lens[2]].unsqueeze(0),
                 value[2,:input_lens[2]].unsqueeze(0))
    loss_3=torch.dist(output_all[2,:input_lens[2]].unsqueeze(0),output_3)
    loss_3_ = torch.dist(output_error[2, :input_lens[2]].unsqueeze(0), output_3)
    print(loss_3, loss_3_)
    torch.save([output_all,output_1,output_2,output_3],'exp_2/exp_2_outputs.pt')

