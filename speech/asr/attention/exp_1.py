import torch
from torch import nn
from attention import MultiHeadedAttention

B=1
T=10
D=16
H=4

exp_1_inputs=torch.load('exp_1/exp_1_inputs.pt')
mha = MultiHeadedAttention(n_head=H, n_feat=D, dropout_rate=0)
mha.load_state_dict(exp_1_inputs['mha'])
query=exp_1_inputs['query']
key=exp_1_inputs['key']
value=exp_1_inputs['value']
output=mha(query,key,value)
torch.save(output,'exp_1/exp_1_outputs.pt')

