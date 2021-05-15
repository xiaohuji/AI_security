import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import accuracy_score

def xgbMultiTrain(X_train, X_val, y_train, y_val, test, num_round):
    # multi-cls model
    xgb_params_multi = {'objective': 'multi:softprob',
                        'num_class': 8,
                        'eta': 0.04,
                        'max_depth': 6,
                        'subsample': 0.9,
                        'colsample_bytree': 0.7,
                        'lambda': 2,
                        'alpha': 2,
                        'gamma': 1,
                        'scale_pos_weight': 20,
                        'eval_metric': 'mlogloss',
                        'silent': 0,
                        'seed': 149}
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(test)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(xgb_params_multi,
                      dtrain,
                      num_round,
                      evals=watchlist,
                      early_stopping_rounds=100,
                      verbose_eval=100
                      )
    p_val = pd.DataFrame(model.predict(dval, ntree_limit=model.best_iteration), index=X_val.index)
    p_test = pd.DataFrame(model.predict(dtest, ntree_limit=model.best_iteration), index=test.index)
    return model, p_val, p_test

def main(ovr_n, n_round):
    train_data = pd.read_csv('./data/train_features_all.csv')
    test_data = pd.read_csv('./data/test_features_all.csv')

    tr_X = train_data.drop(['file_id', 'label'], axis=1)
    tr_y = train_data['label']
    print('[TRAIN FEATURE SIZE]: ', tr_X.shape)
    print('[TRAIN LABEL DISTRIBUTION]: ')
    print(tr_y.value_counts())

    te_X = test_data.drop(['file_id', 'label'], axis=1)
    te_y = test_data['label']
    print('[TEST FEATURE SIZE]: ', te_X.shape)
    print('[TEST LABEL DISTRIBUTION]: ')
    print(te_y.value_counts())

    # plt.figure(figsize=[10, 8])
    # sns.heatmap(train_data.iloc[:1600, 1:12].corr())
    # plt.show()

    print('5-Fold Multi-Class Model Training')
    # Variables
    logloss_rlt = []
    p_val_all = pd.DataFrame()
    p_test_all = pd.DataFrame(np.zeros((te_X.shape[0], 8)))
    model = ''
    # Start 5-fold CV
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    for fold_i, (tr_index, val_index) in enumerate(skf.split(tr_X, tr_y)):
        print('FOLD -', fold_i, ' Start...')
        # Prepare train, val dataset
        X_train, X_val = tr_X.iloc[tr_index, :], tr_X.iloc[val_index, :]
        y_train, y_val = tr_y[tr_index], tr_y[val_index]
        # Train model
        model, p_val, p_test = xgbMultiTrain(X_train, X_val, y_train, y_val, te_X, n_round)
        # Evaluate Model and Concatenate Val-Prediction
        print('--------------------------------------------------')
        print(y_val)
        print(p_val)
        m_log_loss = log_loss(y_val, p_val)
        print('----------------log_loss : ', m_log_loss, ' ---------------------')
        logloss_rlt = logloss_rlt + [m_log_loss]
        truth_prob_df = pd.concat([y_val, p_val], axis=1)
        p_val_all = pd.concat([p_val_all, truth_prob_df], axis=0)
        # Predict Test Dataset
        p_test_all = p_test_all + 0.2 * p_test
    print(type(p_test_all))
    print('Evaluation')
    print('[LOGLOSS]: ', logloss_rlt)
    print('[LOGLOSS MEAN]: ', log_loss(p_val_all.iloc[:, 0], p_val_all.iloc[:, 1:]))
    acc_p_test = np.argmax(p_test_all.values, axis=1)
    accuracy = accuracy_score(te_y.values, acc_p_test)
    print('[ACCURACY]: ', accuracy)
    # print('[LOGLOSS STD]: ', np.std(logloss_rlt))
    print('[LOGLOSS STD]: ', pd.Series(logloss_rlt).std())
    feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
    print('[TOP20 IMPORTANT FEATURES(5TH-FOLD MODEL)]: ')
    print(feat_imp[0:20])
    # g = sns.barplot(x=feat_imp.index[0:20], y=feat_imp[0:20])
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # plt.show()
    print('SUBMIT CHECK')
    rlt = pd.concat([test_data['file_id'], p_test_all], axis=1)
    prob_list = ['prob' + str(i) for i in range(ovr_n)]
    rlt.columns = ['file_id'] + prob_list
    check_flag = all(rlt.iloc[:, 1:].sum(axis=1) - 1 < 1e-6)
    if check_flag:
        print('RESULT IS OK...')
        rlt.to_csv('./submit/rlt_TEST.csv', index=None)
        print('RESULT SAVED...')
    else:
        print('RESULT IS WRONG!')


if __name__ == '__main__':
    main(8, 2000)