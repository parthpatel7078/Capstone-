from flask import Flask, render_template, flash, request, url_for, redirect, session,send_file
from distutils.log import debug
from fileinput import filename
from werkzeug.utils import secure_filename
from flask import *
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
import requests
import json
import pandas as pd
import time
import traceback
from PIL import Image
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
from tensorflow.keras import models, layers
import os
import PIL
import pathlib
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import io
from io import StringIO
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = os.path.join('static', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__,template_folder='templates', static_folder='static')
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'You Will Never Guess'


def init():
    global model,graph,model1
    model=load_model('brain_tumor_detection.h5',compile=False)
    model1=load_model('lung_cancer_detection.h5',compile=False)
    model2=load_model('skin_cancer_detection.h5',compile=False)
    graph=tf.compat.v1.get_default_graph()



def brain_class(img):
    image = preprocessing.image.load_img(img)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    Dict1={0:'glioma_tumor',1:'meningioma_tumor',2:'no_tumor',3:'pituitary_tumor'}
    with graph.as_default():
        #optimizer = Adam(learning_rate=1e-3)
        model=load_model('brain_tumor_detection.h5',compile=False)
        Optimizer=Adam
        #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),optimizer=Optimizer,metrics=['accuracy'])
        pred = model.predict(scaled_img)
        print(pred)
        #probability=probability.max()
        #print(probability)
        y_pred = np.argmax(pred,axis=1)
        print(y_pred)
        label_act = np.array([Dict1[x] for x in y_pred])
    return label_act

def lung_class(img):
    image = preprocessing.image.load_img(img)
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    Dict1={0:'Bengin cases',1:'Malignant cases',2:'Normal cases'}
    with graph.as_default():
        #optimizer = Adam(learning_rate=1e-3)
        model=load_model('lung_cancer_detection.h5',compile=False)
        Optimizer=Adam
        #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),optimizer=Optimizer,metrics=['accuracy'])
        pred = model.predict(scaled_img)
        print(pred)
        #probability=probability.max()
        #print(probability)
        y_pred = np.argmax(pred,axis=1)
        print(y_pred)
        label_act = np.array([Dict1[x] for x in y_pred])
    return label_act

def skin_class(img):
    image = preprocessing.image.load_img(img,target_size=(32,32))
    image_array = preprocessing.image.img_to_array(image)
    scaled_img = np.expand_dims(image_array, axis=0)
    Dict1={0:'akiec',1:'bcc',2:'bkl',3:'df',4:'mel',5:'nc',6:'vasc'}
    with graph.as_default():
        #optimizer = Adam(learning_rate=1e-3)
        model=load_model('skin_cancer_detection.h5',compile=False)
        Optimizer=Adam
        #model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),optimizer=Optimizer,metrics=['accuracy'])
        pred = model.predict(scaled_img)
        print(pred)
        #probability=probability.max()
        #print(probability)
        y_pred = np.argmax(pred,axis=1)
        print(y_pred)
        label_act = np.array([Dict1[x] for x in y_pred])
    return label_act

@app.route('/')
def index():
    return render_template('Final.html')

#@app.route('/facebook')
#def facebook():
 #   return render_template('home.html')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index.html')
 
@app.route('/show_image')
def displayImage():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_data_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image.html', user_image = img_file_path)


@app.route('/cancer_class')
def cancerClass():
    # Get uploaded csv file from session as a json value
    img_file_path = session.get('uploaded_data_file_path', None)
    uploaded_class = brain_class(img_file_path)
    
    
    return render_template('show_image.html',data_var=uploaded_class)

@app.route('/uploadfile12',  methods=("POST", "GET"))
def uploadFile12():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index1.html')

@app.route('/show_image12')
def displayImage12():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_data_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image1.html', user_image = img_file_path)

@app.route('/cancer_class12')
def cancerClass12():
    # Get uploaded csv file from session as a json value
    img_file_path = session.get('uploaded_data_file_path', None)
    uploaded_class = lung_class(img_file_path)
    
    
    return render_template('show_image1.html',data_var=uploaded_class)

@app.route('/uploadfile13',  methods=("POST", "GET"))
def uploadFile13():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index2.html')

@app.route('/show_image13')
def displayImage13():
    # Retrieving uploaded file path from session
    img_file_path = session.get('uploaded_data_file_path', None)
    # Display image in Flask application web page
    return render_template('show_image2.html', user_image = img_file_path)

@app.route('/cancer_class13')
def cancerClass13():
    # Get uploaded csv file from session as a json value
    img_file_path = session.get('uploaded_data_file_path', None)
    uploaded_class = skin_class(img_file_path)
    
    
    return render_template('show_image2.html',data_var=uploaded_class)

if __name__ == "__main__":
    init()
    app.run(debug = True)