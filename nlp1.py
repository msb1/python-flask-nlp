
import re
import spacy
import atexit
import eventlet
import html
from spacy import displacy
from spacy.matcher import PhraseMatcher

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


class Text(object):

        def __init__(self, socketio, text):
            self.socketio = socketio
            self.nlp = spacy.load('en_core_web_md')
            self.matcher = None
            self.rawtext = text
            self.doc = self.nlp(self.rawtext)
            print("__init__", self.doc)
            self.matches = {}
            self.initFlag = False
            self.runFlag = False
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
