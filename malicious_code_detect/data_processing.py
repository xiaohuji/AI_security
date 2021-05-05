import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def read_train_file(path, nrows):
    labels = []
    files = []
    data = pd.read_csv(path, nrows=nrows)
    # for data in data1:
    # 不同的文件,文件13888个（以文件编号汇总）
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        print(file_name, file_group)
        # 获取label
        file_labels = file_group['label'].values[0]
        print('file_labels:',file_labels)
        # 根据线程和顺序排列
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        labels.append(file_labels)
        files.append(api_sequence)
    print(len(labels))
    print(len(files))
    with open(path.split('/')[-1] + ".txt", 'w') as f:
        for i in range(len(labels)):
            f.write(str(labels[i]) + ' ' + files[i] + '\n')

def read_test_file(path, n):
    names = []
    files = []
    data = pd.read_csv(path, nrows=n)
    # for data in data1:
    goup_fileid = data.groupby('file_id')
    for file_name, file_group in goup_fileid:
        print(file_name)
        # file_labels = file_group['label'].values[0]
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        # labels.append(file_labels)
        names.append(file_name)
        files.append(api_sequence)
    print(len(names))
    print(len(files))
    with open("security_test.csv.pkl", 'wb') as f:
        pickle.dump(names, f)
        pickle.dump(files, f)
    # with open(path.split('/')[-1] + ".txt", 'w') as f:
    #     for i in range(len(names)):
    #         f.write(str(names[i]) + ' ' + files[i] + '\n')


def load_train2h5py(path="security_train.csv.txt"):
    labels = []
    files = []
    with open(path) as f:
        for i in f.readlines():
            i = i.strip('\n')
            labels.append(i[0])
            files.append(i[2:])
    labels = np.asarray(labels)
    print(labels.shape)
    with open("security_train.csv.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(files, f)

# 分析各类病毒各api调用数
def ana_top_n(path, nrows, n):
    data = pd.read_csv(path, nrows=nrows)
    # goup_fileid = data.groupby('file_id')
    # process = data.groupby(['file_id', '']).api.count()
    # print(process)
    # process_set = set(process)
    # print(len(process_set))
    # print(process_set)
    # data.groupby('file_id')['label'].value_counts().unstack().plot(kind='bar')
    # print(type(data))
    feat = data.groupby(['label', 'api'])['api'].count().reset_index(name='val')
    # print(feat, type(feat))
    # feat = feat.sort_values('val', ascending=False).groupby(['label']).reset_index()
    feat.groupby('label', group_keys=False).apply(lambda x: x.sort_values('val', ascending=False)).groupby('label').head(n).set_index(['api','label']).plot(kind='bar', figsize=(28, 8), color=['orange'])
    # img = feat.groupby('label').val.plot(kind='bar', figsize=(20, 8))
    # img = feat.groupby(['label', 'val']).head(5).plot(kind='bar', figsize=(20, 8), color=['r'])
    # plt.xticks(rotation=-60)
    plt.show()
    # feat = feat.groupby(['label', 'val'])
    # feat = feat.groupby('label', group_keys=False).apply(lambda x: x.sort_values('val', ascending=False)).groupby('label').head(5).set_index('label')
    print(feat)
    # pd.DataFrame(feat).to_csv("top_five_api.csv", mode='a',header=False)

# 分析各类病毒使用api种类数
def ana_top_api(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat = data.groupby(['label', 'api'])['api'].count().reset_index(name='val')
    img = feat.groupby('label', group_keys=False).val.count()
    feat = feat.groupby('label', group_keys=False).val.count().reset_index(name='val')
    for a, b in zip(feat.iloc[:, 0], feat.iloc[:, 1]):  ##控制标签位置
        plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=14)
    plt.bar(kind='bar', figsize=(10, 8), color=['r'])
    xticks = [0,1,2,3,4,5,6,7]
    xticklabes = ['Normal', 'Ransom', 'Mining', 'DDoS', 'Worm', 'Infection', 'Backdoor', 'Trojan']
    plt.xticks(xticks, xticklabes, rotation=-0)
    # feat = data.groupby(['label']).api.count().plot(kind='bar', figsize=(10, 8), color=['orange'])
    plt.show()
# 分析各类病毒所占数据集比例
def ana_label_ratio_paint(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat = data.groupby(['file_id','label']).api.count().reset_index(name='val')
    print(feat)
    feat = feat.groupby(['label']).val.count().reset_index(name='val', drop=False)
    print(feat)
    print(feat.iloc[:, 0], feat.iloc[:, 1])
    exp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    xticklabes = ['Normal', 'Ransom', 'Mining', 'DDoS', 'Worm', 'Infection', 'Backdoor', 'Trojan']
    plt.pie(x=feat.iloc[:, 1], labels=xticklabes, explode=exp, autopct="%0.2f%%")
    plt.show()
# --------------------------------------------------------------------------------------------------
# tid_distinct_cnt: file发起了多少线程;
# api_distinct_cnt: file调用了多少不同的API ;
# pd.Series.nunique唯一值的统计次数
def ana_label_tid_api(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    return_data = data[['file_id', 'label']].drop_duplicates()
    feat = data.groupby(['file_id']).agg(
        {'tid': pd.Series.nunique, 'api': pd.Series.nunique}).reset_index()
    # print(feat)
    feat.columns = ['file_id', 'tid_distinct_cnt', 'api_distinct_cnt']
    # print(feat)
    return_data = return_data.merge(feat, on='file_id', how='left')
    return return_data

# tid_api_cnt_max, tid_api_cnt_min, tid_api_cnt_mean: ","file中的线程调用的最多 / 最少 / 平均 api数目;
def ana_label_tidapi_max_min_mean (path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat_tmp = data.groupby(['file_id', 'tid']).agg({'index': pd.Series.count, 'api': pd.Series.nunique}).reset_index()
    # print(feat_tmp)
    feat = feat_tmp.groupby(['file_id']).index.agg(['max', 'min', 'mean']).reset_index()
    # print(feat)
    feat.columns = ['file_id', 'tid_api_cnt_max', 'tid_api_cnt_min', 'tid_api_cnt_mean']
    feat = feat_tmp.groupby(['file_id'])['api'].agg(['max', 'min', 'mean']).reset_index()
    feat.columns = ['file_id', 'tid_api_distinct_cnt_max', 'tid_api_distinct_cnt_min', 'tid_api_distinct_cnt_mean']
    return_data = feat.merge(feat, on='file_id', how='left')
    return return_data
# ------------------下面是用透视表的--------------------------------------------------------------------
#每个api第一次调用的Index
def ana_label_api_first_index(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat = data.groupby(['file_id', 'api'])['index'].min().reset_index(name='val')
    # print(feat)
    # 透视表
    feat = feat.pivot(index='file_id', columns='api', values='val')
    # print(feat)
    feat.columns = [feat.columns[i] + '_index_min' for i in range(feat.shape[1])]
    feat_withFileid = feat.reset_index()
    # print(feat_withFileid)
    return feat_withFileid

# 统计api调用的次数
def ana_label_api_num(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat = data.groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
    feat = feat.pivot(index='file_id', columns='api', values='val')
    feat.columns = [feat.columns[i] + '_cnt' for i in range(feat.shape[1])]
    feat_withFileid = feat.reset_index()
    return feat_withFileid


# 统计api调用的比例
def ana_label_api_ratio(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    feat = ana_label_api_num(path, nrows)
    tmp = data.groupby(['file_id']).api.count()
    feat_rate = pd.concat([feat, tmp], axis=1)
    feat_rate = feat_rate.apply(lambda x: x / feat_rate.api)
    feat_rate.columns = [feat_rate.columns[i] + '_rate' for i in range(feat_rate.shape[1])]
    feat_rate_withFileid = feat_rate.reset_index().drop(['api_rate'], axis=1)
    return feat_rate_withFileid
# ---------------------------------TF-IDF-OVR-NB-LR------------------------------------------
def tfidfModelTrain(train_path, test_path, train_n, test_n):
    train = pd.read_csv(train_path, nrows=train_n)
    test = pd.read_csv(test_path, nrows=test_n)
    api_vec = TfidfVectorizer(ngram_range=(1, 4),
                              min_df=3, max_df=0.9,
                              strip_accents='unicode',
                              use_idf=1, smooth_idf=1, sublinear_tf=1)
    tr_api = train.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
    te_api = test.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
    tr_api_vec = api_vec.fit_transform(tr_api['api'])
    val_api_vec = api_vec.transform(te_api['api'])
    return tr_tfidf_rlt, te_tfidf_rlt
# NB-LR
def pr(x, y_i, y):
    # 按列相加
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def get_mdl(x, y):
    y = y.values
    # 朴素贝叶斯
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    x_nb = x.multiply(r)
    np.random.seed(0)
    m = LogisticRegression(C=6, dual=True, random_state=0)
    return m.fit(x_nb, y), r


def nblrTrain(tr_tfidf_rlt, te_tfidf_rlt, train, ovr_n):
    label_fold = []
    preds_fold_lr = []
    preds_te = np.zeros((te_tfidf_rlt.shape[0], ovr_n))
    lr_oof_tr = pd.DataFrame()
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    for fold_i, (tr_index, val_index) in enumerate(skf.split(train, train['label'])):
        if fold_i >= 0:
            tr, val = train.iloc[tr_index], train.iloc[val_index]
            x = tr_tfidf_rlt[tr_index, :]
            test_x = tr_tfidf_rlt[val_index, :]
            preds = np.zeros((len(val), ovr_n))
            preds_te_i = np.zeros((te_tfidf_rlt.shape[0], ovr_n))
            labels = [i for i in range(ovr_n)]
            for i, j in enumerate(labels):
                print('fit', j)
                m, r = get_mdl(x, tr['label'] == j)
                preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
                preds_te_i[:, i] =preds_te_i[:, i] + m.predict_proba(te_tfidf_rlt.multiply(r))[:, 1]

            preds_lr = preds
            lr_oof_i = pd.DataFrame({'file_id': val['file_id']})
            for i in range(ovr_n):
                lr_oof_i['prob' + str(i)] = preds[:, i]
            lr_oof_tr = pd.concat([lr_oof_tr, lr_oof_i], axis=0)

            # 归一化后 测试一下
            for i, j in enumerate(preds_lr):
                preds_lr[i] = j / sum(j)
            log_loss_i = log_loss(val['label'], preds_lr)
            print('log_loss_i', log_loss_i)
            # label_fold.append(val['label'].tolist())
            # preds_fold_lr.append(preds_lr)
    # 五折后顺序是乱的
    lr_oof_tr = lr_oof_tr.sort_values('file_id')
    preds_te_avg = preds_te / 5
    lr_oof_te = pd.DataFrame({'file_id': range(0, te_tfidf_rlt.shape[0])})
    for i in range(ovr_n):
        lr_oof_te['prob' + str(i)] = preds_te_avg[:, i]

    return lr_oof_tr, lr_oof_te


if __name__ == '__main__':
    path = './security_train/security_train.csv'
    path1 = './security_train/security_train.csv'
    path2 = './security_test/security_test.csv'
    # read_train_file(path1, 200000)
    # read_test_file(path2, 200000)
    # load_train2h5py(path="security_train.csv.txt")

    # ana_top_n(path, 20000000, 5)
    # ana_top_api(path, 10000000)

    train_1 = ana_label_tid_api(path, 200000)
    train_2 = ana_label_tidapi_max_min_mean(path, 200000)
    test_1 = ana_label_tid_api(path2, 200000)
    test_2 = ana_label_tidapi_max_min_mean(path2, 200000)

    train = train_1.merge(train_2, on=['file_id'], how='left')
    test = test_1.merge(test_2, on=['file_id'], how='left')

    train_3 = ana_label_api_first_index(path, 200000)
    train_4 = ana_label_api_num(path, 200000)
    train_5 = ana_label_api_ratio(path, 200000)
    train_pi = train_3.merge(train_4, on=['file_id'], how='left')
    train_pi = train_pi.merge(train_5, on=['file_id'], how='left')

    test_3 = ana_label_api_first_index(path2, 200000)
    test_4 = ana_label_api_num(path2, 200000)
    test_5 = ana_label_api_ratio(path2, 200000)
    test_pi = test_3.merge(test_4, on=['file_id'], how='left')
    test_pi = test_pi.merge(test_5, on=['file_id'], how='left')

    interaction_feat = train_pi.columns[train_pi.columns.isin(test_pi.columns.values)].values
    train_pi = train_pi[interaction_feat]
    test_pi = test_pi[interaction_feat]
    train = train.merge(train_pi, on=['file_id'], how='left')
    test = test.merge(test_pi, on=['file_id'], how='left')

    tr_tfidf_rlt, te_tfidf_rlt = tfidfModelTrain(path, path2, 200000, 200000)
    lr_oof_tr, lr_oof_te = nblrTrain(tr_tfidf_rlt, te_tfidf_rlt, train_1, ovr_n=1)

    # 这个ovr_n要改
    prob_list = ['prob' + str(i) for i in range(1)]
    train = pd.concat(
        [train, lr_oof_tr[prob_list]], axis=1)
    test = pd.concat(
        [test, lr_oof_te[prob_list]], axis=1)

    print('[TRAIN SIZE]: ', train.shape)
    print('[TEST SIZE]: ', test.shape)

    train.to_csv('/data/train_features_all.csv', index=None)
    test.to_csv('/data/test_features_all.csv', index=None)

    tr_X = train.drop(['`file_id', 'label'], axis=1)
    tr_y = train['label']
    print('[TRAIN FEATURE SIZE]: ', tr_X.shape)
    print('[TRAIN LABEL DISTRIBUTION]: ')
    print(tr_y.value_counts())

    te_X = test.drop(['file_id', 'label'], axis=1)
    te_y = test['label']
    print('[TEST FEATURE SIZE]: ', te_X.shape)
    print('[TEST LABEL DISTRIBUTION]: ')
    print(te_y.value_counts())
