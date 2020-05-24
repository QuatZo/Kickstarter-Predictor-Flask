from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import time
import shutil

import requests
import json
import os

from joblib import load
import xgboost as xgb

import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_prediction():
    data = request.get_json()
    df = pd.DataFrame(pd.Series(data['data'])).T

    req = requests.get('http://puma.swstats.info/' + data['key'])

    if '.zip' in data['key']:
        model_name = 'model' + str(time.time()) + '.zip'
        open(model_name, 'wb').write(req.content)
        shutil.unpack_archive(model_name, data['key'][7:-4])
        model = tf.keras.models.load_model(data['key'][7:-4])
        os.remove(model_name)
        shutil.rmtree(data['key'][7:-4], ignore_errors=True)
    else:
        model_name = 'model' + str(time.time()) + '.bin'
        open(model_name, 'wb').write(req.content)
        model = load(model_name)
        os.remove(model_name)

    try:
        if '.zip' in data['key']:
            raise AttributeError # force predict function instead of predict_proba since it will be deprecated in future Tensorflow 
        prediction = [float(number) for number in model.predict_proba(df)[0]]
        probability = True
    except AttributeError:
        prediction = int(model.predict(df)[0])
        probability = False

    data['proba'] = probability
    data['prediction'] = prediction

    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)