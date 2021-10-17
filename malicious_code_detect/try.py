# import pandas as pd
# import numpy as np
# import torch
# flag = torch.cuda.is_available()
# print(flag)
# import pdb
# data = pd.read_csv('./security_train/try.csv')
# feat = data.groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
# print(feat)
# feat = data.groupby(['file_id', 'api'])['api'].count().reset_index(name='val')
# print(feat)
# feat = data.groupby(['file_id', 'api'])['label'].count().reset_index(name='val')
# # print(feat)
#
# import tensorflow as tf
# print(tf.test.is_gpu_available())
# for i, j in enumerate(labels):
#     print('fit', i,j)
# data={"two":np.linspace(1,4,4)}
# df=pd.DataFrame(data,index=[1,2,3,4])
# data2={"two":np.linspace(1,4,4)}
# df2=pd.DataFrame(data,index=[1,2,3,4])
# y = df['two']==1
# print(y.values)
# print(df)
# print((y.values == 1).sum())
wordlist = ['1','2','3','4','5']
dic = dict([(w,1) for w in wordlist])
print(dic)
# wordlist = [[1],[2],[3],[4]]
for i in range(len(wordlist)):
    if wordlist[i] == '\n':
        wordlist = wordlist[i:]
        break
wordlist = ''.join(''.join(wordlist).strip().split())
print(wordlist)
print(wordlist.count('8'))
