from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
import requests
import json
import facebook
import pandas as pd
import time
import traceback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def sentiment_scores(data_frame):
    sentiment_list = []
    for row_num in range(len(data_frame)):
        sentence = data_frame['fb_comments'][row_num]
    

IMAGE_FOLDER=os.path.join('static','Img_pool')
app=Flask(__name__)

app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}

def init():
    global model,graph
    model=load_model('Final_Model.h5')
    graph=tf.compat.v1.get_default_graph()
    
@app.route('/',methods=['GET','POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction(data_frame):
    if request.method=='POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        #text = request.form['text']
        #print(text)
        #comments=text
        #input_data = np.column_stack([comments])
        #X = pd.DataFrame(data=input_data,columns=['comments'])
        texts = df['fb_comments'].values
        Sentiment = ''
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        x_test_tokens = tokenizer.texts_to_sequences(texts)
        x_test_pad = pad_sequences(x_test_tokens, maxlen=46,padding='pre', truncating='pre')
        #print(x_test_pad)
        #max_review_length = 500
        #word_to_id = imdb.get_word_index()
        #strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        #text = text.lower().replace("<br />", " ")
        #text=re.sub(strip_special_chars, "", text.lower())

        #words = text.split() #split string into a list
        #x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        #x_test = sequence.pad_sequences(x_test, maxlen=500) # Should be same which you used for training data
        vector = np.array([x_test_pad.flatten()])
        print(vector)
        #model.compile(run_eagerly=True)
        with graph.as_default():
            
            optimizer = Adam(lr=1e-3)
            model=load_model('Final_Model.h5')
            model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
            probability = model.predict(array([vector][0]))
            print(probability)
            probability=probability.max()
            print(probability)
            class1 = np.argmax(probability)
            print(class1)
        if class1== 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad_image.jpg')
        elif class1== 1:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy_image.jpg')
        else:
            sentiment = 'Neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neutral_image.png')
    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.run()