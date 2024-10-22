import torch
from typing import Tuple, List
from collections import defaultdict
import math
from ctc import ctc_prefix_beam_search,ctc_greedy_search,get_dict


if __name__ =='__main__':

    dict_path = '/root/Desktop/speech/asr/ctc/dict.txt'
    index2word = get_dict(dict_path)    # 实现索引向字符的映射

    print('ctc_prefix_beam_search开始解码')
    hyps,scores=ctc_prefix_beam_search(encoder_out='encoder_out.pt',
                                       beam_size=3)
    print('ctc_prefix_beam_search完成解码')
    output=" ".join([index2word[i] for i in hyps])
    print(f'score:{scores:.4f}\t解码结果:{output}\n', )

    print('ctc_greedy_search开始解码')
    hyps,scores=ctc_greedy_search(encoder_out='encoder_out.pt')
    print('ctc_greedy_search完成解码')
    output=" ".join([index2word[i] for i in hyps])
    print(f'score:{scores:.4f}\t解码结果:{output}\n', )
