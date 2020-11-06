import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import json
import pprint
import pandas as pd 
from io import BytesIO

class WSGICopyBody(object):
    def __init__(self, application):
        self.application = application

    def __call__(self, environ, start_response):
        length = int(environ.get('CONTENT_LENGTH') or 0)
        body = environ['wsgi.input'].read(length)
        environ['body_copy'] = body
        environ['wsgi.input'] = BytesIO(body)
        return self.application(environ, start_response)

app = Flask(__name__)
model=load_model('gru-model.h5')
app.wsgi_app = WSGICopyBody(app.wsgi_app)

def preprocess(req):
    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(req)
    X = tokenizer.texts_to_sequences(req)
    X=np.asarray(X)
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    req=request.__dict__
    print(req)
    df = pd.DataFrame.from_dict(req) 
    print(df)

    X=preprocess(req)
    prediction = model.predict(X)
    if(prediction>=0.5):
        return render_template('index.html', prediction_text='The request is malicious')
    else:
        return render_template('index.html', prediction_text='The request is not malicious')


if __name__ == "__main__":
    app.run(debug=True)