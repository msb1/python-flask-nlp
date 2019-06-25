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

    print('START READING DATA... elapsed time: {} sec'.format(time.process_time() - start_time))
    # Get the data.

    with open('data\\amazon\\train.csv', mode='r', encoding='utf-8') as fr, open('data\\amazon\\train.dmp', mode='wb') as fw:
        index = 0
        for line in fr:
            items = line.split(',')
            title = ''
            if type(items[1]) == str:
                title = items[1].replace('\\n', '\n').replace('\\"', '"')
            body = ''
            if type(items[2]) == str:
                body = items[2].replace('\\n', '\n').replace('\\"', '"')
            label = int(items[0].replace('\"', ''))
            text = title + ', ' + body

            doc = clean_text(nlp(text))
            tokens = doc.to_array([ID])   
            pickle.dump([label, tokens], fw)
            if index % 10000 == 0:
                print('Training record cleaned with index: {}... elapsed time: {}'.format(index, time.process_time() - start_time))
            index += 1

    print('FINISH CLEANING train.csv... elapsed time: {} sec'.format(time.process_time() - start_time))

    with open('data\\amazon\\test.csv', mode='r', encoding='utf-8') as fr, open('data\\amazon\\test.dmp', mode='wb') as fw:
        index = 0
        for line in fr:
            items = line.split(',')
            title = ''
            if type(items[1]) == str:
                title = items[1].replace('\\n', '\n').replace('\\"', '"')
            body = ''
            if type(items[2]) == str:
                body = items[2].replace('\\n', '\n').replace('\\"', '"')
            label = int(items[0].replace('\"', ''))
            text = title + ', ' + body

            doc = clean_text(nlp(text))
            tokens = doc.to_array([ID])   
            pickle.dump([label, tokens], fw)
            if index % 10000 == 0:
                print('Test record cleaned with index: {}... elapsed time: {}'.format(time.process_time() - start_time, index))
            index += 1

    print('FINISH CLEANING test.csv... elapsed time: {} sec'.format(time.process_time() - start_time))


if __name__ == "__main__":
    main()
