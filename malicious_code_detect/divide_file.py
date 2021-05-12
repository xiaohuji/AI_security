
import os
import pandas as pd

# df = pd.read_csv('./data/security_train.csv',nrows=69935083)
df = pd.read_csv('./data/security_train.csv')
train = df[:69935083]
test = df[69935083:]
train.to_csv('./data/security_train2.csv', index=None)
test.to_csv('./data/security_test.csv', index=None)
# grouped = df.groupby("file_id")
# title = df.columns.values
# print(grouped.group_keys)
# # grouped.values
# # for value, group in grouped:
# #     print(value)
# #     print('222222222222222222222')
# #     print(group)
