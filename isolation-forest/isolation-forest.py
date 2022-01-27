# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:37:11 2022

@author: waasiq
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('./Water_Data.csv')

# Convert the string to pandas datetime
dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) 

# Putting the Data in Missing 
dataset = dataset.fillna(method = 'pad')


sub_dataset = dataset.filter(['timestamp','Turb(FNU)'], axis=1)
t = sub_dataset

X = dataset.iloc[:,:1].values
y = dataset.iloc[:,2:3].values


# Splitting the dataset into the training set and data set

#from sklearn.model_selection import train_test_split
#X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# Fitting the regression model to the dataset
from sklearn.ensemble import IsolationForest
regressor = IsolationForest(contamination = 'auto')
regressor.fit(X)

# Predicting the result with Isolation Forest Method
anamoly = regressor.predict(X)
anomaly_score  = regressor.decision_function(X)


# ----------------------- ALGORITHM FINSHES ----------------------------

# Visualizing the Isolation Forest Results
t['anomaly']=pd.Series(regressor.predict(X))
anomaly_data = t.loc[t['anomaly'] == -1, ['timestamp','Turb(FNU)']]

t = t.fillna(method = 'pad')
anomaly_data = anomaly_data.fillna(method = 'pad')

#print("Percentage of anomalies in data: {:.2f}".format((len(t.loc[t['anomaly']==-1])/len(t))*100))


plt.style.use('seaborn')
plt.figure(figsize=(12,8))

plt.plot(t['timestamp'],t['Turb(FNU)'], color='blue', label = 'Normal')
plt.plot_date(x=anomaly_data['timestamp'],y=anomaly_data['Turb(FNU)'], color='red', label = 'Anomaly')

#plt.plot(predictions, label='Predictions')

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d, %b, %Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.xlabel("Datatime Index ")
plt.ylabel("Turbidity Values(FNU)")
plt.title('Anomaly Detection : Isolation Forest Method')
plt.legend()
plt.show()

