import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Dropout, Embedding, Concatenate
from keras.regularizers import l2


# BiDirectional LSTM
# For Amazon model
def bilstm(embedding_size, embedding_dim, embedding_matrix, record_dim, dropout=0.1, num_class=5, activation='softmax'):
# For Imdb model
# def bilstm(embedding_size, embedding_dim, embedding_matrix, record_dim, dropout=0.3, num_class=1, activation='sigmoid'):
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_dim, weights=[embedding_matrix], input_length=record_dim, trainable=False))
    # for Amazon model
    # model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Bidirectional(LSTM(64)))
    # for Imdb model
    # model.add(Bidirectional(CuDNNLSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
    # model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))
    model.add(Dropout(dropout))
    # for Amazon model
    model.add(Dense(8))
    model.add(Dense(num_class, activation=activation))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model




