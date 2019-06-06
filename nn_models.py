import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Dropout, Embedding, Concatenate
from keras.regularizers import l2


# BiDirectional LSTM
def bilstm(embedding_size, embedding_dim, embedding_matrix, record_dim, dropout=0.3):

    model = Sequential()
    model.add(Embedding(embedding_size, embedding_dim, weights=[embedding_matrix], input_length=record_dim, trainable=False))
    # model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
    # model.add(Dense(1, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


