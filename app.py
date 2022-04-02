#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model_lr = pickle.load(open('model_lr.pkl', 'rb'))

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')

@app.route('/fuel', methods=['POST', 'GET'])
def rfuel():
    return render_template('resultf.html')

@app.route('/resultf.html', methods=['POST', 'GET'])
def fuel():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model_lr.predict(final_features)
    output = prediction[0]
    pred = int(output)
    return render_template('resultf.html', prediction_text='Fuel price for kilometer driven is :{}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)

