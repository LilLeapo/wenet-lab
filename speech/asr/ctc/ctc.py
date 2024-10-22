import torch
from typing import Tuple, List
from collections import defaultdict
import math



def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp

def get_dict(dict_path):
    index2word = {}
    with open(dict_path,'r',encoding='utf8') as f:
        for i in f.readlines():
            temp=i.strip('\n').split(' ')
            if len(temp)==2:
                word,index=temp
                index2word[int(index)]=word
    return index2word

dict_path = '/root/Desktop/speech/asr/ctc/dict.txt'
index2word = get_dict(dict_path)


def ctc_prefix_beam_search(encoder_out='encoder_out',beam_size=3):

    detail=False    # 是否要打印中间过程

    # 1. Encoder forward and get CTC score
    encoder_out=torch.load(encoder_out) # 读取已经保存好的编码输出结果
    maxlen = encoder_out.size(1)
    ctc_probs = encoder_out#torch.log_softmax(encoder_out,dim=-1)  # (1, maxlen, vocab_size)
    ctc_probs = ctc_probs.squeeze(0)
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)

        if detail:
            print()
            print(f'{t:02d}阶段起始序列')
            for cur_hyp in cur_hyps:
                print(f'score:{log_add(cur_hyp[1]):.4f}\t序列:{" ".join([index2word[i] for i in cur_hyp[0]])}', )


            print(f'{t:02d}阶段候选字符')
            for i in range(beam_size):
                print(f'{index2word[top_k_index[i].int().item()]}\t{top_k_logp[i]:.4f}')


        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]

                    ##########################################
                    # 补全代码1:调用log_add函数更新当前阶段序列prefix对应的n_pb
                    n_pb = 
                    ##########################################

                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]

                    ##########################################
                    # 补全代码2:调用log_add函数更新当前阶段序列prefix对应的n_pnb
                    n_pnb = 
                    ##########################################

                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]

                    ##########################################
                    # 补全代码3:调用log_add函数更新当前阶段序列n_prefix对应的n_pnb
                    n_pnb = 
                    ##########################################

                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]

                    ##########################################
                    # 补全代码4:调用log_add函数更新当前阶段序列n_prefix对应的n_pnb
                    n_pnb = 
                    ##########################################

                    next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                           key=lambda x: log_add(list(x[1])),
                           reverse=True)
        cur_hyps = next_hyps[:beam_size]
        if detail:
            print(f'{t:02d}阶段候选序列')
            for cur_hyp in next_hyps:
                print(f'score:{log_add(cur_hyp[1]):.4f}\t序列:{" ".join([index2word[i] for i in cur_hyp[0]])}', )

            print(f'{t:02d}阶段剪枝结果')
            for cur_hyp in cur_hyps:
                print(f'score:{log_add(cur_hyp[1]):.4f}\t序列:{" ".join([index2word[i] for i in cur_hyp[0]])}', )

    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    scores = hyps[0][1]
    hyps = hyps[0][0]
    return hyps,scores

def ctc_greedy_search(encoder_out='encoder_out'):
    encoder_out=torch.load(encoder_out) # 读取已经保存好的编码输出结果

    ctc_probs = torch.log_softmax(encoder_out,dim=-1)  # (1, maxlen, vocab_size)
    ctc_probs = ctc_probs.squeeze(0)    # (maxlen, vocab_size)

    topk_prob, topk_index = ctc_probs.topk(1, dim=1)  # (maxlen,1)
    hyps = topk_index.squeeze(1).tolist()
    hyps=[i for i in hyps if i !=0]
    scores = topk_prob.sum()

    return hyps, scores



if __name__ =='__main__':

    hyps,scores=ctc_prefix_beam_search(encoder_out='encoder_out.pt',beam_size=3)
    print('ctc_prefix_beam_search完成解码')
    print(f'score:{scores:.4f}\t解码结果:{" ".join([index2word[i] for i in hyps])}\n', )


    hyps,scores=ctc_greedy_search(encoder_out='encoder_out.pt')
    print('ctc_greedy_search完成解码')
    print(f'score:{scores:.4f}\t解码结果:{" ".join([index2word[i] for i in hyps])}', )


