# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:59:57 2022

@author: YUSUF
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mpl_dates

# Data upload
dataset = pd.read_csv('..\data\custom-dataset.csv')

parameters = dataset.iloc[:,0:5].values
Potability = dataset.iloc[:,5:6].values

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(parameters,Potability,test_size=0.2, random_state=0)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)


if list(y_pred).count(False) > list(y_test).count(False):
   print("Accuracy true False:", (list(y_test).count(False)/list(y_pred).count(False)) * 100)
   FP = list(y_test).count(False)/list(y_pred).count(False)
else:
    print("Accuracy true False:", (list(y_pred).count(False)/list(y_test).count(False)) * 100)
    FP = list(y_pred).count(False)/list(y_test).count(False)

if list(y_pred).count(True) > list(y_test).count(True):
    print("Accuracy true True:", (list(y_test).count(True)/list(y_pred).count(True)) * 100)
    TP = list(y_test).count(True)/list(y_pred).count(True)
    
else:
    print("Accuracy true True:", (list(y_pred).count(True)/list(y_test).count(True)) * 100)
    TP = list(y_pred).count(True)/list(y_test).count(True)
    
topVeriSayisi = list(y_test).count(True) + list(y_test).count(False) 
positive = list(y_pred).count(True)*TP + list(y_pred).count(False)*FP
print("Accuracy:",positive/topVeriSayisi*100)