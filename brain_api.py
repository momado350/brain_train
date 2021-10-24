#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:06:35 2018

@author: vivekkalyanarangan
"""

#import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd
from joblib import load

model = load('./br.joblib') 

#with open('./rf.pkl', 'rb') as model_file:
#    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_brain():
    """Example endpoint returning a prediction of iris
    ---
    parameters:
      - name: weight
        in: query
        type: number
        required: true
      
    """
    weight = request.args.get("weight")
    
    
    prediction = model.predict(np.array([[weight]]))
    return str(prediction)

@app.route('/predict_file', methods=["POST"])
def predict_brain_file():
    """Example file endpoint returning a prediction of iris
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    prediction = model.predict(input_data)
    return str(list(prediction))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    