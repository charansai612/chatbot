#!/usr/bin/env python

# importing required libaries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

# read data
df = pd.read_excel('./data/chat_bot.xlsx')
print("Reading Data")

# dropping null
df.dropna(inplace=True)
print("Dropping NaN")

#initialze lemmatizer
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

print("In the Halfway....")

# applying
df['lemmatized'] = df['Context'].apply(text_normalize)

#initialze
cv = CountVectorizer()

#bow
df_lema = df['lemmatized']

X = cv.fit_transform(df_lema).toarray()

# tfidf
tfidf = TfidfVectorizer()

def we_tfidf(lemmatized):
    X_tfid = tfidf.fit_transform(lemmatized).toarray()
    return X_tfid

# applying tfidf
x_tfid = we_tfidf(df.lemmatized)

print("Preprocessing done")

# query flittering
def query(string):
    #string = input("Enter Query:")
    clean = text_normalize(string)
    clean_bow = tfidf.transform([clean]).toarray()
    return clean_bow

#ask query
print("Test Bot")
query_ask = query("Hello")

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

# Reply
validation(x_tfid,query_ask)

print("Done Training...")

from flask import Flask, render_template, request, url_for

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods=['POST'])
def process():
    user_input = request.form['user_input']
    query_ask = query(user_input)
    response = validation(x_tfid,query_ask)
    return render_template('index.html',user_input=user_input,bot_response=response)

if __name__=='__main__':
	app.run(debug=True)