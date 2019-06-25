'''
Created on May 6, 2019

@author: Barnwaldo
'''

import json
import eventlet
import time
import json
from flask import Flask, request, render_template, redirect, url_for
from flask_socketio import SocketIO
from nlp1 import Text


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.threaded = True
# app.config['DEBUG'] = True
socketio = SocketIO(app)
plaintext = ""
textobj = None


@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    print(">> rendering index.html...")
    return render_template('index.html')


@app.route('/sidenav')
def sidenav():
    print(">> sending sidenav.html to jquery request...")
    return render_template('sidenav.html')


@app.route('/topmenu')
def topmenu():
    print(">> sending topmenu.html to jquery request...")
    return render_template('topmenu.html')


@socketio.on('connect', namespace='/comm')
def comm_connect():
    print('>> comm socketIO connected')


@socketio.on('connect', namespace='/text')
def text_connect():
    print('>> text socketIO connected')


@socketio.on('message', namespace='/comm')
def comm_message(msg):
    global plaintext, textobj
    print(">> message received on comm socketIO: ", msg) 
    obj = json.loads(msg) 
    if obj['msgType'] == 'getFileText':
        fpath = obj['filepath'] + obj['inFile']
        try:
            with open(fpath) as f:
                plaintext = f.read()
            print(plaintext)
            socketio.emit('message', {'type': 'text', 'payload': plaintext}, namespace='/text') 
        except:
            print('Error: File does not exist or path is entered incorrectly...')
    
    elif obj['msgType'] == 'matcher':
        if textobj is None: 
            print("Matcher called without text...")
            return
        print(obj['phrase1'], obj['phrase2'], obj['phrase3'])
        phrases = [ obj['phrase1'], obj['phrase2'], obj['phrase3']]
        match = textobj.phrase_matcher(phrases)
        print (match)
        socketio.emit('message', {'type': 'match', 'payload': match}, namespace='/text')

    elif obj['msgType'] == 'sentiment':
        if textobj is None: 
            print("Sentiment called without text...")
            return
        sentiment = textobj.sentiment()
        print (sentiment)
        socketio.emit('message', {'type': 'sentiment', 'payload': sentiment}, namespace='/text')

    elif obj['msgType'] == 'similarity':
        if textobj is None: 
            print("Similarity called without initial text for comparison...")
            return
        # print(obj['compareText'])
        similarity = textobj.similarity(obj['compareText'])
        print (similarity)
        socketio.emit('message', {'type': 'similarity', 'payload': similarity}, namespace='/text')

        
@socketio.on('message', namespace='/text')
def text_message(msg):
    global plaintext, textobj
    print(">> message received on text socketIO: ", msg) 
    obj = json.loads(msg) 
    if obj['msgType'] == 'saveText':
        fpath = obj['filepath'] + obj['outFile']
        plaintext = obj['plaintext']
        textobj = Text(socketio, plaintext)
        # make Spacy doc from plaintext
        doc = textobj.doc
        # print('TOKENS:')
        # make list of json token objects for bootstrap table       
        token_list = textobj.tokenize()
        socketio.emit('message', {'type': 'token', 'payload': json.dumps(token_list)}, namespace='/text') 

        # print('ENTITIES:', len(list(doc.ents)))
        ent_list = textobj.entity_extraction()
        socketio.emit('message', {'type': 'entity', 'payload': json.dumps(ent_list)}, namespace='/text')

        # print('NOUN CHUNKS:')
        noun_list = textobj.noun_chunks()
        socketio.emit('message', {'type': 'noun', 'payload': json.dumps(noun_list)}, namespace='/text')

        entity_table = textobj.entity_display()
        # print(entity_table)
        socketio.emit('message', {'type': 'entityDisplay', 'payload': entity_table}, namespace='/text')


@socketio.on('disconnect', namespace='comm')
def comm_disconnect():
    print('>> comm socketIO disconnected')


@socketio.on('disconnect', namespace='text')
def text_disconnect():
    print('>> text socketIO disconnected')


if __name__ == '__main__':
    socketio.run(app)
