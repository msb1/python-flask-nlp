
import re
import spacy
import atexit
import eventlet
import html
import numpy as np
import spacy
from spacy import displacy
from spacy.tokens import Doc
from spacy.attrs import ID, LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.matcher import PhraseMatcher
from keras.models import load_model


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
REPLACE_BR = re.compile("(</br>)")
REPLACE_DIV_ENT = re.compile('<div class=\"entities\"')
PH_MARK = ['<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">',
           '<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">',
           '<mark class="entity" style="background: #aa9cfcm ; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">']
MARK_CLOSE = '</mark>'
SPAN = '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">'
SPAN_CLOSE = '</span>'
SENTIMENTS = ["SUPER NEGATIVE", "JUST NEGATIVE", "ALMOST NEUTRAL", "KINDA POSITIVE", "REALLY POSITIVE"]

class Text(object):

        def __init__(self, socketio, text):
            self.socketio = socketio
            self.nlp = spacy.load('en_core_web_lg')
            self.matcher = None
            self.rawtext = text
            self.doc = self.nlp(self.rawtext)
            self.id_array = None
            print("__init__", self.doc)
            self.matches = {}
            self.initFlag = False
            self.runFlag = False
            self.model1 = None
            self.model2 = None
            atexit.register(self.cleanup)

        
        def cleanup(self):
            pass

        
        # basic preprocessing to remove html (currently in the javascript as well but want it available on backend here)
        # def preprocess_text(self, text):
        #     text = [REPLACE_NO_SPACE.sub("", line.lower()) for line in text]
        #     text = [REPLACE_WITH_SPACE.sub(" ", line) for line in text]

        
        def tokenize(self):
            print(self.doc)
            token_list = []
            for token in self.doc:
                token_dict = {}
                if token.pos == 'PUNCT': continue
                # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.vector_norm, token.is_alpha, token.is_stop,token.has_vector, token.is_oov)
                token_dict['text']  = token.text
                token_dict['lemma']  = token.lemma_
                token_dict['pos']  = token.pos_
                token_dict['tag']  = token.tag_
                token_dict['dep']  = token.dep_
                token_dict['shape']  = str(token.shape_)
                token_dict['vector_norm'] = str(token.vector_norm)
                token_list.append(token_dict)
            return token_list


        def entity_extraction(self):
            ent_list = []
            for ent in self.doc.ents:
                ent_dict = {}
                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
                ent_dict['text']  = ent.text
                ent_dict['startChar']  = ent.start_char
                ent_dict['endChar']  = ent.end_char
                ent_dict['label']  = ent.label_
                ent_list.append(ent_dict)
            return ent_list


        def noun_chunks(self):
            noun_list = []
            for chunk in self.doc.noun_chunks:
                noun_dict = {}
                # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
                noun_dict['text']  = chunk.text
                noun_dict['rootText']  = chunk.root.text
                noun_dict['rootDep']  = chunk.root.dep_
                noun_dict['rootHeadText']  = chunk.root.head.text
                noun_list.append(noun_dict)
            return noun_list

        
        def entity_display(self):
            htmlstr = displacy.render(self.doc, style="ent")
            htmlstr = REPLACE_BR.sub(" ", htmlstr)
            # htmlstr = REPLACE_DIV_ENT.sub('<div class=\"form-control\" id=\"entityTable\"', htmlstr)
            return htmlstr


        def phrase_matcher(self, phrases):
            self.matcher = PhraseMatcher(self.nlp.vocab)
            patterns = [self.nlp(phrase) for phrase in phrases]
            # print(patterns)
            self.matcher.add("PH", None, *patterns)
            html_match = '<div class="entities" style="line-height: 2.5; direction: ltr">'
            html_match += 'Phrases: '
            for i in range(len(phrases)):
                ind = phrases.index(phrases[i])
                html_match += PH_MARK[i]  + SPAN + phrases[i] + SPAN_CLOSE + MARK_CLOSE 
            html_match += '<br/>'    
            current = 0
            for match_id, start, end in self.matcher(self.doc):
                html_match += str(self.doc[current:start])
                phrase = str(self.doc[start:end])
                ind = phrases.index(phrase)
                print(ind, phrase)
                html_match += PH_MARK[ind] + SPAN + phrase + SPAN_CLOSE + MARK_CLOSE
                current = end
            html_match += str(self.doc[current: -1])
            html_match += '</div>'
            return html_match


        def sentiment(self):
            if self.model1 is None:
                self.model1 = load_model('imdb_bilstm_model.h5')
            if self.model2 is None:
                self.model2 = load_model('amazon_bilstm_model.h5')
            
            self.clean_text()
            tok_length = len(self.id_array)

            # imdb model
            MAX_LEN = 500   # from imdb training
            tokens = np.copy(self.id_array)
            if tok_length < MAX_LEN:
                tokens = np.concatenate((np.zeros(MAX_LEN - tok_length), tokens), axis=0).astype(np.int32)
            else:
                tokens = tokens[tok_length - MAX_LEN:].astype(np.int32)
            tokens = np.array([tokens])
            sent1_score = self.model1.predict(tokens, batch_size=1, verbose=1)
            sent1 = self.model1.predict_classes(tokens)
            sentiment1 = ["POSITIVE" if sent1[0][0] == 1 else "NEGATIVE"]
            print(sent1_score, sent1, sentiment1)
            print("Sentiment Analysis -- Imdb Training Set: score = {:.2f} predicting class {}". format(sent1_score[0][0], sentiment1[0]))

            # amazon model
            MAX_LEN = 100   # from amazon training
            tokens = np.copy(self.id_array)
            if tok_length < MAX_LEN:
                tokens = np.concatenate((np.zeros(MAX_LEN - tok_length), tokens), axis=0).astype(np.int32)
            else:
                tokens = tokens[tok_length - MAX_LEN:].astype(np.int32)
            tokens = np.array([tokens])
            sent2_score = self.model2.predict(tokens, batch_size=1, verbose=1)
            sent2 = self.model2.predict_classes(tokens)
            sentiment2 = SENTIMENTS[sent2[0]]
            sentiment2_scores = "[ "
            for i in range(5):
                sentiment2_scores += "{}: {:.2f}".format((i + 1), sent2_score[0][i]) 
                if i < 4: sentiment2_scores += ", "
            sentiment2_scores += " ]"
            print(sent2_score, sent2, sentiment2)
            print("Sentiment Analysis -- Amazon Training Set: score = {} predicting class {}". format(sentiment2_scores, sentiment2))

            # create html reply for web page display
            html_sentiment = '<div class="entities" style="line-height: 2.5; direction: ltr">'
            html_sentiment += '<ul><li>'
            html_sentiment += "Imdb Training Set:     score = {:.2f}   |   {}". format(sent1_score[0][0], sentiment1[0])
            html_sentiment += '</li><li>'
            html_sentiment += "Amazon Training Set:   score = {}   |   {}". format(sentiment2_scores, sentiment2)
            html_sentiment += '</li></ul></div>'
            return html_sentiment


        def clean_text(self):
            '''
            Remove tokens that are punctuation, numbers, symbols or stop words
            '''
            indices = []
            for index, token in enumerate(self.doc):
                if token.pos_  in ('PUNCT', 'NUM', 'SYM') or token.is_stop:
                    indices.append(index)
            np_array = self.doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
            np_array = np.delete(np_array, indices, axis = 0)
            clean_doc = Doc(self.doc.vocab, words=[t.text for i, t in enumerate(self.doc) if i not in indices])
            clean_doc.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
            self.id_array = self.doc.to_array([ID])
            

        def similarity(self, ctext):
            '''
            Perform cosine similarity with Spacy (large model) bewteen two text
            '''
            cdoc = self.nlp(ctext)
            spacy_similarity = self.doc.similarity(cdoc)

            # create html reply for web page display
            html_similarity = '<div class="entities" style="line-height: 2.5; direction: ltr">'
            html_similarity += '<ul><li>'
            html_similarity += "Spacy score = {:.3f}". format(spacy_similarity)
            # html_similarity += '</li><li>'
            # html_similarity += "Amazon Training Set:   score = {}   |   {}". format(sentiment2_scores, sentiment2)
            html_similarity += '</li></ul></div>'
            return html_similarity