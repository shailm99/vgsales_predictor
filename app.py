#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:58:47 2020

@author: shailmirpuri
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
import category_encoders as ce
app = Flask(__name__)
file_name = 'xgboost_model.pkl'
f=open(file_name, 'rb')
model = pickle.load(f)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

X_train=pd.read_csv('X_train.csv',index_col='Rank')
X_test=pd.read_csv('X_test.csv',index_col='Rank')
y_train=pd.read_csv('y_train.csv',index_col='Rank')
y_test=pd.read_csv('y_test.csv',index_col='Rank')
cat_features= ['Genre', 'Device']       
target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(X_train[cat_features], y_train['Global_Sales'])

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = float(request.form['Year'])
        EU_Sales=float(request.form['EU_Sales'])
        NA_Sales=float(request.form['NA_sales'])
        JP_Sales=float(request.form['JP_Sales'])
        OR_Sales=float(request.form['OR_Sales'])
        Series=request.form['Series']
        Genre=request.form['Genre']
        Device=request.form['Device']
        if(Series=='Fifa'):
            FIFA=1 
            Mario=0
            Call_of_Duty=0 
            Grand_Theft_Auto=0 
            Pokemon=0
            Halo=0
            Wii=0
            NBA=0
        elif(Series=='Mario'):
            FIFA=0 
            Mario=1
            Call_of_Duty=0 
            Grand_Theft_Auto=0 
            Pokemon=0
            Halo=0
            Wii=0
            NBA=0
        elif(Series=='Call of Duty'):
            FIFA=0 
            Mario=0
            Call_of_Duty=1
            Grand_Theft_Auto=0 
            Pokemon=0
            Halo=0
            Wii=0
            NBA=0
        elif(Series=='Grand Theft Auto'):
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=1
            Pokemon=0
            Halo=0
            Wii=0
            NBA=0
        elif(Series=='Pokemon'):
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=0
            Pokemon=1
            Halo=0
            Wii=0
            NBA=0
        elif(Series=='Halo'):
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=0
            Pokemon=0
            Halo=1
            Wii=0
            NBA=0
        elif(Series=='Wii'):
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=0
            Pokemon=0
            Halo=0
            Wii=1
            NBA=0
        elif(Series=='NBA'):
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=0
            Pokemon=0
            Halo=0
            Wii=0
            NBA=1
        else:   
            FIFA=0 
            Mario=0
            Call_of_Duty=0
            Grand_Theft_Auto=0
            Pokemon=0
            Halo=0
            Wii=0
            NBA=0
        lst=[[Genre,Device]]
        t=pd.DataFrame(lst, columns=cat_features)
        t_e=target_enc.transform(t[cat_features])
        D=float(t_e.Device)
        G=float(t_e.Genre)
        a=np.array([[Year,FIFA,Mario,Call_of_Duty,Grand_Theft_Auto,Pokemon,
                                   Halo,Wii,NBA,NA_Sales,EU_Sales,JP_Sales,OR_Sales,G,D]])
        np.reshape(a,(1,-1))
        pred=model.predict(a)
        output=float(pred[0])
        output=round(output,2)
        if output>0:
            return render_template('index.html',prediction_text="The Global Sales Predicted for this game is {} million".format(output))
        else: 
            return render_template('index.html',prediction_text="This game won't be bought")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)