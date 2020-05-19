from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import time

import requests
import json
import os
import time

from joblib import load
import xgboost as xgb

app = Flask(__name__)

@app.route('/', methods=['POST'])
def get_prediction():
    data = request.get_json()
    df = pd.DataFrame(pd.Series(data['data'])).T

    req = requests.get('http://puma.swstats.info/' + data['key'])

    model_name = 'model' + str(time.time()) + '.bin'
    open(model_name, 'wb').write(req.content)
    model = load(model_name)
    os.remove(model_name)

    try:
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