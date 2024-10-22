import os
import sys
uttid_wav_text=sys.argv[1]
dev_rate=float(sys.argv[2])
save_data_path=sys.argv[3]
random=(sys.argv[4]=='true')

with open(uttid_wav_text,'r',encoding='utf8') as f:
    uttid_wav_text=[]
    for i in f.readlines():
        i=i.strip('\n').split('\t')
        if len(i)==3:
            uttid_wav_text.append(i)

from random import shuffle
shuffle(uttid_wav_text)

data_num=len(uttid_wav_text)
dev_data_num=int(data_num*dev_rate)
test_data_num=int(data_num*dev_rate)
train_data_num=data_num-dev_data_num-test_data_num
print(data_num,train_data_num,dev_data_num,test_data_num,)

if os.path.exists(f'{save_data_path}/train'):
    pass
else:
    os.makedirs(f'{save_data_path}/train')
train_data=uttid_wav_text[:train_data_num]
with open(f'{save_data_path}/train/wav.scp','w',encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[1]}\n' for i in train_data])
with open(f'{save_data_path}/train/text', 'w', encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[2]}\n' for i in train_data])

if os.path.exists(f'{save_data_path}/dev'):
    pass
else:
    os.makedirs(f'{save_data_path}/dev')
dev_data=uttid_wav_text[train_data_num:-test_data_num]
with open(f'{save_data_path}/dev/wav.scp','w',encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[1]}\n' for i in dev_data])
with open(f'{save_data_path}/dev/text', 'w', encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[2]}\n' for i in dev_data])


if os.path.exists(f'{save_data_path}/test'):
    pass
else:
    os.makedirs(f'{save_data_path}/test')
test_data=uttid_wav_text[-test_data_num:]
with open(f'{save_data_path}/test/wav.scp','w',encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[1]}\n' for i in test_data])
with open(f'{save_data_path}/test/text', 'w', encoding='utf8') as f:
    f.writelines([f'{i[0]} {i[2]}\n' for i in test_data])






