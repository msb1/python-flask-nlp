import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlp.vocab)
nyc = "New York City"
ul = "urban legend"
ss = "subway system"
patterns = [nlp(nyc), nlp(ul), nlp(ss)]
print(patterns)
text = "Rats in New York City are widespread, as they are in many densely populated areas. For a long time, the number of rats in New York City was unknown, and a common urban legend declared there were up to five times as many rats as people. In 2014, however, scientists more accurately measured the entire city's rat population to be approximately only 24% of the number of humans. That would reduce the urban legend's ratio considerably, with approximately 2 million rats to New York's 8.4 million people at the time of the study. Even the city's subway system has had problems."
matcher.add("NYC", None, *patterns)
doc = nlp(text)
matches = matcher(doc)
for match_id, start, end in matches:
    print(nlp.vocab.strings[match_id], start, end, doc[start:end])

######################################################################
# Some Flask Form Code for Phrase Matching - currently using SocketIO

# @app.route('/matcher', methods=['GET', 'POST'])
# def matcher():
#     print("matcher form request")
#     print(request.form['phrase1'], request.form['phrase2'], request.form['phrase3'])
#     if request.method == "POST":
#         p1 = request.form['phrase1']
#         p2 = request.form['phrase2']
#         p3 = request.form['phrase3']
#         print(p1, p2, p3)
#         phrase_dict = {'phrase1':p1, 'phrase2':p2, 'phrase3':p3}
#     textobj.init_phrase_matcher(phrase_dict)
#     mm = textobj.phrase_matcher()
#     matches = []
#     for m in mm:
#         entry = {'id': m.match_id, 'phrase': phrase_dict[m.match_id], 'text': textobj.doc[m.start: m.end]}
#         matches.append(entry)
#     print(matches)
#     print(">> rendering matcher.html...")
#     # return render_template('matcher.html', results)
#     return render_template('matcher.html', matches=matches)


######################################################################
# Some code to show sentences and vectors from Spacy
        # print('SENTS')
        # for sent in list(doc.sents):
        #     print(sent)
        # print('VECTOR')
        # for vector in list(doc.vector):
        #     print("____________")
        #     print(vector)
        # try:
        #     with open(fpath, 'w') as f:
        #         filetext = f.write(plaintext)
        # except:
        #     print('Error: File write problem or path is entered incorrectly...')