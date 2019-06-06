import time
import pickle
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, LOWER, POS, ENT_TYPE, IS_ALPHA
from load_data import load_imdb_sentiment_analysis_dataset, load_amazon_reviews_sentiment_analysis_dataset

MAX_LEN = 500   # max text length - truncate records or pad to this length

def clean_text(doc):
    '''
    Remove tokens that are punctuation, numbers, symbols or stop words
    '''
    indices = []
    for index, token in enumerate(doc):
        if token.pos_  in ('PUNCT', 'NUM', 'SYM') or token.is_stop:
            indices.append(index)
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    np_array = np.delete(np_array, indices, axis = 0)
    clean_doc = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indices])
    clean_doc.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    return clean_doc


def main():
    '''
    Read training and validation text from files and clean text data sets (remove punctuation, symbols and stop words)
    '''
    # initialize timer
    start_time = time.process_time()

    # initialize spacy language with pretrained model
    nlp = spacy.load('en_core_web_lg')
    spacy.vocab.link_vectors_to_models(nlp.vocab)

    print('READING DATA... elapsed time: {} sec'.format(time.process_time() - start_time))
    # Get the data.
    data = load_amazon_reviews_sentiment_analysis_dataset('data\\amazon')
    # data = load_imdb_sentiment_analysis_dataset('data\\imdb')
    (raw_train_texts, train_labels), (raw_val_texts, val_labels) = data

    # output cleaned label data to  files with binary pickle serialization
    with open('amazon_train_labels.dmp', 'wb') as fp:
    # with open('imdb_train_labels.dmp', 'wb') as fp:
        pickle.dump(train_labels, fp)

    with open('amazon_val_labels.dmp', 'wb') as fp:
    # with open('imdb_val_labels.dmp', 'wb') as fp:
        pickle.dump(val_labels, fp)

    print('PREPROCESSING DATA - PUNCTUATION CLEANING... elapsed time: {} sec'.format(time.process_time() - start_time))
    # clean training and validation reviews
    # tokenize with current vector space index
    train_tokens = []
    index = 0
    for text in raw_train_texts:
        doc = clean_text(nlp(text))
        train_tokens.append(doc.to_array([ID]))
        print('Training record cleaned with index: ', index)
        index += 1
        
    val_tokens = []
    for text in raw_val_texts:
        doc = clean_text(nlp(text))
        val_tokens.append(doc.to_array([ID]))
        print('Validation record cleaned with index: ', index)
        index += 1

    print('PREPROCESSING DATA - COMPLETE... elapsed time: {} sec'.format(time.process_time() - start_time))

    # output cleaned text data to  files with binary pickle serialization
    with open('amazon_train.dmp', 'wb') as fp:
    # with open('imdb_train.dmp', 'wb') as fp:
        pickle.dump(train_tokens, fp)

    with open('amazon_val.dmp', 'wb') as fp:
    # with open('imdb_val.dmp', 'wb') as fp:
        pickle.dump(val_tokens, fp)

    print('PICKLING - COMPLETE... elapsed time: {} sec'.format(time.process_time() - start_time))

    with open ('amazon_train.dmp', 'rb') as fp:
    # with open ('imda_train.dmp', 'rb') as fp:
        train_tokens = pickle.load(fp)
    print(train_tokens[0])
    print(train_tokens[100])
    print(train_tokens[500])

    with open ('amazon_val.dmp', 'rb') as fp:
    # with open ('imdb_val.dmp', 'rb') as fp:
        val_tokens = pickle.load(fp)
    print(val_tokens[0])
    print(val_tokens[100])
    print(val_tokens[500])

if __name__ == "__main__":
    main()