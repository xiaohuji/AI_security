import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Activation, Input, Lambda, concatenate, Reshape, LSTM, RNN, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from tensorflow.keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time
import numpy as np
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


maxlen = 600

# def mulitl_version_lstm():
#     embed_size = 256
#     num_filters = 64
#     kernel_size = [3, 5, 7]
#     main_input = Input(shape=(maxlen,))
#     emb = Embedding(304, 256, input_length=maxlen)(main_input)
#     # _embed = SpatialDropout1D(0.15)(emb)
#     warppers = []
#     warppers2 = []  # 0.42
#     warppers3 = []
#     for _kernel_size in kernel_size:
#         conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(emb)
#         warppers.append(AveragePooling1D(2)(conv1d))
#     for (_kernel_size, cnn) in zip(kernel_size, warppers):
#         conv1d_2 = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(cnn)
#         warppers2.append(AveragePooling1D(2)(conv1d_2))
#     for (_kernel_size, cnn) in zip(kernel_size, warppers2):
#         conv1d_2 = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(cnn)
#         warppers3.append(AveragePooling1D(2)(conv1d_2))
#     fc = Maximum()(warppers3)
#     rl = LSTM(512)(fc)
#     main_output = Dense(8, activation='softmax')(rl)
#     model = Model(inputs=main_input, outputs=main_output)
#     return model


def Build():
    main_input = Input(shape=(maxlen,), dtype='float64')
    embedder = Embedding(304, 256, input_length=maxlen)
    embed = embedder(main_input)
    # avg = GlobalAveragePooling1D()(embed)
    # cnn1模块，kernel_size = 3
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(embed)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(cnn1)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    conv1_1 = Conv1D(64, 3, padding='same', activation='relu')(cnn1)

    conv1_2 = Conv1D(64, 3, padding='same', activation='relu')(conv1_1)

    cnn1 = MaxPool1D(pool_size=2)(conv1_2)
    rl = LSTM(64)(cnn1)
    # flat = Flatten()(cnn3)
    # drop = Dropout(0.5)(flat)
    fc = Dense(64)(rl)

    main_output = Dense(8, activation='softmax')(rl)
    model = Model(inputs=main_input, outputs=main_output)
    return model

def main(train_path, test_path, train_n, test_n):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('GPU: ', gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # session = tf.compat.v1.keras.backend.setSession(config=config)

    Fname = 'malware_'
    Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    tensorboard = TensorBoard(log_dir='./Logs/' + Time, histogram_freq=0, write_graph=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    train = pd.read_csv(train_path, nrows=train_n)
    test = pd.read_csv(test_path, nrows=test_n)
    outfiles = test.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()['api'].tolist()
    outlabels = test[['file_id', 'label']].drop_duplicates().label.tolist()
    files = train.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()['api'].tolist()
    labels = train[['file_id', 'label']].drop_duplicates().label.tolist()
    # enumerate用
    labels_d = labels.copy()




    # with open("wordsdic.pkl", 'rb') as f:
    #     tokenizer = pickle.load(f)
    #

    labels = np.asarray(labels)
    # print(labels)
    # 将类别向量转换为二进制
    labels = to_categorical(labels, num_classes=8)
    # print(labels)

    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                          split=' ',
                          char_level=False,
                          oov_token=None,
                          lower=False)
    tokenizer.fit_on_texts(files)
    tokenizer.fit_on_texts(outfiles)
    # print(tokenizer.word_docs)

    # with open("wordsdic.pkl", 'wb') as f:
    #     pickle.dump(tokenizer, f)

    # vocab = tokenizer.word_index
    # print(vocab)
    # print(len(vocab))
    x_train_word_ids = tokenizer.texts_to_sequences(files)
    x_out_word_ids = tokenizer.texts_to_sequences(outfiles)
    # print(x_train_word_ids)
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=maxlen)
    x_out_padded_seqs = pad_sequences(x_out_word_ids, maxlen=maxlen)
    # print(x_train_padded_seqs)
    meta_train = np.zeros(shape= (len(x_train_padded_seqs), 8))
    meta_test = np.zeros(shape=(len(x_out_padded_seqs), 8))
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    for i, (tr_ind, te_ind) in enumerate(skf.split(x_train_padded_seqs, labels_d)):
        print('FOLD: {}'.format(str(i)))
        print(len(te_ind), len(tr_ind))
        X_train, X_train_label = x_train_padded_seqs[tr_ind], labels[tr_ind]
        X_val, X_val_label = x_train_padded_seqs[te_ind], labels[te_ind]

        model = Build()
        # model = load_model('model_weight.h5')
        print(model.summary())
        # exit()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model_save_path = './model/model_weight_cnn_lstm_{}.h5'.format(str(i))
        print(model_save_path)
        print('i: ', i)
        if i in [-1]:
            model.load_weights(model_save_path)
            print(model.evaluate(X_val, X_val_label))
        else:
            # 每个epoch保存一次
            checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=True)
            ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', baseline=None,
                                restore_best_weights=False)
            history = model.fit(X_train, X_train_label,
                                batch_size=128,
                                epochs=100,
                                validation_data=(X_val, X_val_label), callbacks=[checkpoint, ear])

            # model.save('./model/model_weight_cnn_lstm_{}.h5'.format(str(i)))
            model.load_weights(model_save_path)
            # model = load_model('model_weight.h5')
        pred_val = model(X_val)
        pred_test = model(x_out_padded_seqs)

        meta_train[te_ind] = pred_val
        meta_test += pred_test
        K.clear_session()

    meta_test /= 5.0
    print('meta_test: ', meta_test)
    acc_p_test = np.argmax(meta_test, axis=1)
    accuracy = accuracy_score(outlabels, acc_p_test)
    print('[ACCURACY]: ', accuracy)
    meta_test.to_csv('./submit/lstm_rlt_TEST.csv', index=None)
    # with open("cnn_lstm_result.csv", 'wb') as f:
    #     pickle.dump(meta_train, f)
    #     pickle.dump(meta_test, f)
if __name__ == '__main__':
    path = './security_train/security_train.csv'
    path1 = './security_train/security_train.csv'
    path2 = './security_train/security_test.csv'
    main(path, path, 2000000, 200000)

    # result = model.predict(x_out_padded_seqs)
    # out = []
    # for i in range(len(file_names)):
    #     tmp = []
    #     a = result[i].tolist()
    #     # for j in range(len(a)):
    #     #     a[j] = ("%.5f" % a[j])
    #
    #     tmp.append(file_names[i])
    #     tmp.extend(a)
    #     out.append(tmp)
    # with open("result_lstm.csv", "w", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     # 先写入columns_name
    #     writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
    #                      ])
    #     # 写入多行用writerows
    #     writer.writerows(out)


# def mulitl_version_lstm():
#     embed_size = 256
#     num_filters = 64
#     kernel_size = [3, 5, 7]
#     main_input = Input(shape=(maxlen,))
#     emb = Embedding(304, 256, input_length=maxlen)(main_input)
#     _embed = SpatialDropout1D(0.15)(emb)
#     warppers = []
#     warppers2 = []
#     for _kernel_size in kernel_size:
#         conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(_embed)
#         warppers.append(MaxPool1D(2)(conv1d))
#     for (_kernel_size, cnn) in zip(kernel_size, warppers):
#         conv1d_2 = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation='relu', padding='same')(cnn)
#         warppers2.append(MaxPool1D(2)(conv1d_2))
#     fc = Add()(warppers2)
#     rl = CuDNNLSTM(512)(fc)
#     main_output = Dense(8, activation='softmax')(rl)
#     model = Model(inputs=main_input, outputs=main_output)
#     return model