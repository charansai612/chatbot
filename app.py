# importing required libaries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
import pickle

# read data
df = pd.read_excel('./data/chat_bot.xlsx')
print("Reading Data")

# initalize lemmatizer
lemmatizer = WordNetLemmatizer()

# text_normalization
def text_normalize(text):
    lema_sent = []
    try:
        pre_text = text.lower()
        words = re.sub(r'[^a-z0-9 ]','',pre_text)
        tag_list = pos_tag(nltk.word_tokenize(words),tagset=None)
        for token,pos_token in tag_list:
            if pos_token.startswith('V'): #verb
                pos_val = 'v'
            elif pos_token.startswith('J'): #adjective
                pos_val = 'a'
            elif pos_token.startswith('R'): #adverb
                pos_val = 'r'
            else: #any parts of speech except verb, adjective, adverb
                pos_val = 'n'
            lema_token = lemmatizer.lemmatize(token,pos_val) #computing lematization
            lema_sent.append(lema_token) #append values in list
            
        return " ".join(lema_sent)
    except:
        pass

# cross validation and reply
def validation(x_tfid,query_ask):
    cos = 1 - pairwise_distances(x_tfid,query_ask,metric='cosine')
    ind = cos.argmax()
    threshold = cos[ind]
    if threshold > 0.2:
        result =  df['Text Response'].loc[ind]
    else:
        result =  df['Text Response'].loc[51]
    
    return result

# load pickel
tfidf = pickle.load(open("model/tfidf.pkl","rb"))
x_tfid = pickle.load(open("model/x_tfidf.pkl","rb"))

#make flask ready
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods=['GET', 'POST'])
def process():
    user_input = request.form['user_input']
    query_ask = text_normalize(user_input)
    query_ask = tfidf.transform([query_ask]).toarray()
    response = validation(x_tfid,query_ask)
    return render_template('index.html',user_input=user_input,bot_response=response)