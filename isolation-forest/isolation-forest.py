# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:37:11 2022

@author: waasiq

"""

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset = pd.read_csv('../data/custom-dataset.csv')

X = dataset.iloc[:,:-1].values # other variables - to send to model
y = dataset.iloc[:,5:].values  # potability

#--------------------------- Start of test data split ----------------------

# Splitting the dataset into the training set and data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#--------------------------- End of Test Data Split -----------------------

# Fitting the IsolationForest model to the dataset
from sklearn.ensemble import IsolationForest
model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', random_state = 42)
model.fit(X_train)

# Predicting the result with Isolation Forest Method
y_pred = model.predict(X_test)
anomaly_score  = model.decision_function(X_test)

if list(y_pred).count(-1) > list(y_test).count(False):
   print("Accuracy true False(False Negative):", (list(y_test).count(False)/list(y_pred).count(-1)) * 100)
   FP = list(y_test).count(False)/list(y_pred).count(-1)
else:
    print("Accuracy true False(False Negative):", (list(y_pred).count(-1)/list(y_test).count(False)) * 100)
    FP = list(y_pred).count(-1)/list(y_test).count(False)

if list(y_pred).count(1) > list(y_test).count(True):
    print("Accuracy true True(True Positive):", (list(y_test).count(True)/list(y_pred).count(1)) * 100)
    TP = list(y_test).count(True)/list(y_pred).count(1)
else:
    print("Accuracy true True(True Positive):", (list(y_pred).count(1)/list(y_test).count(True)) * 100)
    TP = list(y_pred).count(1)/list(y_test).count(True)

topVeriSayisi = list(y_test).count(True) + list(y_test).count(False) 
positive = list(y_pred).count(1)*TP + list(y_pred).count(-1)*FP
print("Accuracy: ", positive/topVeriSayisi*100)