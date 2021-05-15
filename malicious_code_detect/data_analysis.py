import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
import time
import csv
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score


# with open("security_train.csv.pkl", "rb") as f:
#     labels = pickle.load(f)
#     files = pickle.load(f)

# with open("security_test.csv.pkl", "rb") as f:
#     file_names = pickle.load(f)
#     outfiles = pickle.load(f)

train = pd.read_csv('./data/security_train2.csv', nrows=10000000)
test = pd.read_csv('./data/security_test.csv', nrows=19000000)
api_vec = TfidfVectorizer(ngram_range=(1, 4),
                          min_df=3, max_df=0.9,
                          strip_accents='unicode',
                          use_idf=1, smooth_idf=1, sublinear_tf=1)
tr_api = train.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
te_api = test.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
print('te_api', te_api)
train_features = api_vec.fit_transform(tr_api['api'])
# print(train_features)
out_features = api_vec.transform(te_api['api'])
labels = train[['file_id', 'label']].drop_duplicates()
labels = labels['label'].values
# print(labels)
te_labels = test[['file_id', 'label']].drop_duplicates()
s = te_labels
te_labels = te_labels['label'].values

print("start tfidf...")
# 稀疏矩阵按词典顺序来
# vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9, )

# 标准化
# train_features = vectorizer.fit_transform(files)
# print("train_features",train_features)
# out_features = vectorizer.transform(outfiles)
# print("out_features",out_features)
# with open("tfidf_feature_no_limit.pkl", 'wb') as f:
#     pickle.dump(train_features, f)
#     pickle.dump(out_features, f)
#
# with open("tfidf_feature_no_limit.pkl", 'rb') as f:
#     train_features = pickle.load(f)
#     out_features = pickle.load(f)
print(train_features.shape)
print(out_features.shape)
meta_train = np.zeros(shape=(len(labels), 8))
meta_test = np.zeros(shape=(len(te_labels), 8))
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
bst = ''
for i, (tr_ind, te_ind) in enumerate(skf.split(train_features, labels)):
    print(tr_ind)
    print(type(train_features))
    print(labels[tr_ind])
    X_train, X_train_label = train_features[tr_ind], labels[tr_ind]
    X_val, X_val_label = train_features[te_ind], labels[te_ind]
    print('X_train', X_train)
    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind))
    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    print("dtrain:", X_train)
    print("dlabel:", X_train_label)
    dtest = xgb.DMatrix(X_val, label=X_val_label)
    dout = xgb.DMatrix(out_features)
    param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 0, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 0.8,
             'colsample_bytree': 0.85}  # 参数

    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 100  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # dtr = xgb.DMatrix(train_features)
    pred_val = bst.predict(dtest)
    print("pred_val:", pred_val)
    pred_test = bst.predict(dout)
    print("pred_test", pred_test)
    meta_train[te_ind] = pred_val
    meta_test += pred_test
meta_test /= 5.0
# print(meta_test)
result = meta_test
from sklearn.preprocessing import OneHotEncoder
print(te_labels)
print(meta_test)

y_true = label_binarize(te_labels, classes=[0,1,2,3,4,5,6,7])

loss = log_loss(y_true, meta_test)
print('log_loss',loss)
test_x = np.argmax(meta_test, axis=1)
# for i in range(len(meta_test)):
#         max_value=max(meta_test[i])
#         test_x.append(meta_test[i].index(max_value))
print('test_x ', test_x)
print('right_x', te_labels)
accuracy_score = accuracy_score(te_labels, test_x)
print('accuracy_score',accuracy_score)

feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
print('[TOP20 IMPORTANT FEATURES(5TH-FOLD MODEL)]: ')
print(feat_imp[0:20])
# print(result)
out = []
for i in range(len(te_labels)):
    tmp = []
    a = result[i].tolist()
    # for j in range(len(a)):
    #     a[j] = ("%.5f" % a[j])
    # print(file_names[i])
    tmp.append(i)
    tmp.extend(a)
    out.append(tmp)



with open("./mulltimodel_xgd_boost_tf{}.csv".format(
        str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))),
        "w",
        newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)
# with open("tfidf_result.pkl", 'wb') as f:
#     pickle.dump(meta_train, f)
#     pickle.dump(meta_test, f)
#     # preds = bst.predict(dout)
