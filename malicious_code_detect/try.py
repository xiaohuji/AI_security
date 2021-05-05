import pandas as pd
import numpy as np
# import torch
# flag = torch.cuda.is_available()
# print(flag)

# data = pd.read_csv('./security_train/try.csv')
# feat = data.groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
# print(feat)
# feat = data.groupby(['file_id', 'api'])['api'].count().reset_index(name='val')
# print(feat)
# feat = data.groupby(['file_id', 'api'])['label'].count().reset_index(name='val')
# # print(feat)
labels = [i for i in range(1)]
print(labels)
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