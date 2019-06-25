import numpy as np
import plotly
import plotly.graph_objs as go
import time
import pickle
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, LOWER, POS, ENT_TYPE, IS_ALPHA
from sklearn.model_selection import train_test_split

import  tensorflow  as tf

from nn_models import bilstm
from clean import clean_text

MAX_LEN = 500
BATCH_SIZE = 50
USE_CUDA = True
LEARN_RATE = 0.1
MOMENTUM = 0.9
EPOCHS = 10
SAVE_MODEL = False
LOG_INTERVAL = 1


def plot(epochs, history):
    xdata = list(range(1, epochs + 1))
    trace1 = go.Scatter(
                    x = xdata,
                    y = history['acc'],
                    name='Training Accuracy',
                    line=dict(color='green'))

    trace2 = go.Scatter(
                    x = xdata,
                    y = history['val_acc'],
                    name='Validation Accuracy',
                    line=dict(color='blue'))

    trace3 = go.Scatter(
                    x = xdata,
                    y = history['loss'],
                    name='Training Loss',
                    line=dict(color='red'))

    trace4 = go.Scatter(
                    x = xdata,
                    y = history['val_loss'],
                    name='Validation_Loss',
                    line=dict(color='orange'))

    layout = go.Layout(
        showlegend=True,
        xaxis = dict(
            title='Epoch'),
        yaxis=dict(
            title='Value'
            ))

    # can show manual ticks with tickvals = []

    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data,layout=layout)

    plotly.offline.plot(fig, filename='TextClass.html', auto_open=True)


def main():
    # initialize timer
    start_time = time.process_time()

    # initialize spacy language with pretrained model
    nlp = spacy.load('en_core_web_lg')
    spacy.vocab.link_vectors_to_models(nlp.vocab)

    # read labels and cleaned data from pickled files
    with open ('imdb_train_labels.dmp', 'rb') as fp:
        train_labels = pickle.load(fp)

    with open ('imdb_val_labels.dmp', 'rb') as fp:
        val_labels = pickle.load(fp)

    with open ('imdb_train.dmp', 'rb') as fp:
        train_tokens = pickle.load(fp)

    with open ('imdb_val.dmp', 'rb') as fp:
        val_tokens = pickle.load(fp)
    
    print('READ PICKLED DATA FROM FILES... elapsed time: {} sec'.format(time.process_time() - start_time))

    # pad token lists with zeros (whitespace) or truncate at beginning
    for i in range(len(train_tokens)):
        tokens = train_tokens[i]
        tok_length = len(tokens)
        if tok_length < MAX_LEN:
            train_tokens[i] = np.concatenate((np.zeros(MAX_LEN - tok_length), tokens), axis=0).astype(np.int32)
        else:
            train_tokens[i] = tokens[tok_length - MAX_LEN:].astype(np.int32)

    train_tokens = np.array(train_tokens)

    for i in range(len(val_tokens)):
        tokens = val_tokens[i]
        tok_length = len(tokens)
        if tok_length < MAX_LEN:
            val_tokens[i] = np.concatenate((np.zeros(MAX_LEN - tok_length), tokens), axis=0).astype(np.int32)
        else:
            val_tokens[i] = tokens[tok_length - MAX_LEN:].astype(np.int32)

    val_tokens = np.array(val_tokens)

    print('PREPROCESSED DATA - TRUNCATION AND PADDING... elapsed time: {} sec'.format(time.process_time() - start_time))

    embedding_index = {}
    for key, vector in nlp.vocab.vectors.items():
        row = nlp.vocab.vectors.find(key=key) 
        word = nlp.vocab.strings[key]
        embedding_index[word] = row
        # print(key, nlp.vocab.strings[key], row, vector)

    embedding_matrix = nlp.vocab.vectors.data
    embedding_shape = embedding_matrix.shape
    print('Embedding Matrix shape:', embedding_shape)
    print('Embedding Index length:', len(embedding_index))

    print('EMBEDDING MATRIX CREATED... elapsed time: {} sec'.format(time.process_time() - start_time))

    model = bilstm(embedding_shape[0], embedding_shape[1], embedding_matrix, MAX_LEN)
    history = model.fit(train_tokens, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_tokens, val_labels))

    # model.save('imdb_bilstm_model.h5')

    print('TRAINING AND VALIDATION COMPLETE... elapsed time: {} sec'.format(time.process_time() - start_time))
    plot(EPOCHS, history.history)

    # text = "Next time you want to visit a world-class art museum, consider Bentonville, Arkansas. Crystal Bridges Museum of American Art is a new must-see in the art world. Its collection includes paintings, sculpture, photography, drawing, and installation art from colonial times to today and features American artists such as Maya Lin, Andrew Wyeth, Jacob Lawrence, Jackson Pollock, and Georgia O’Keeffe. But perhaps the biggest piece of art is the museum itself. Designed by Moshe Safdie, the building blends seamlessly and beautifully into the surrounding Ozark landscape. In addition to the art and architecture collections, there are eight nature trails on the grounds. And admission to the museum is always free."
    # doc = clean_text(nlp(text))
    # temp = doc.to_array([ID])
    # temp_len = len(temp)

    # if temp_len < MAX_LEN:
    #     temp = np.concatenate((np.zeros(MAX_LEN - temp_len), temp), axis=0).astype(np.int32)
    # else:
    #     temp = temp[temp_len - MAX_LEN:].astype(np.int32)


    # print("Model prediction", model.predict(np.array([temp]), batch_size=1))
    # print("Model prediction classes", model.predict_classes(np.array([temp])))

if __name__ == "__main__":
    main()