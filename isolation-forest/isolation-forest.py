# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:37:11 2022

@author: waasiq
"""

# Libraries
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import pandas as pd
import numpy as np

from dataparser import readFile

dataset,X,y = readFile()


#--------------------------- Start of test data split ----------------------

# Splitting the dataset into the training set and data set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_test = X_test.astype('float64')


#--------------------------- End of Test Data Split -----------------------

# Fitting the IsolationForest model to the dataset
from sklearn.ensemble import IsolationForest
model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', random_state = 42)
model.fit(X_train)

# Predicting the result with Isolation Forest Method
anamoly = model.predict(X_test)
anomaly_score  = model.decision_function(X_test)

outcome = pd.DataFrame(data = y_test, index = range(20), columns = ['timestamp'] )
outcomeAnamoly = pd.DataFrame(data = anamoly, index = range(20), columns = ['anamoly'] )

outcomeConcat = pd.concat([outcome , outcomeAnamoly], axis = 1)
graph_plotter = outcomeConcat

# ----------------------- Graph Visualization Below -----------------------

# Visualizing the Isolation Forest Results anamoly w.r.t to Time
graph_plotter['anomaly']= pd.Series(anamoly)
anomaly_data = graph_plotter.loc[graph_plotter['anomaly'] == -1, ['timestamp','anamoly']]


# Prints the percentage anomalies in data
print("Accuracy:", (list(anamoly).count(1)/anamoly.shape[0]) * 100)
print("Percentage of anomalies in data: {:.2f}".format((len(graph_plotter.loc[graph_plotter['anomaly']==-1])/len(graph_plotter))*100))

plt.style.use('seaborn')
plt.figure(figsize=(12,8))

plt.plot(graph_plotter['timestamp'],graph_plotter['anamoly'], color='blue', label = 'Normal')
plt.plot_date(x=anomaly_data['timestamp'],y=anomaly_data['anamoly'], color='red', label = 'Anomaly')

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d, %b, %Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.xlabel("Datatime Index")
plt.ylabel("anamoly")
plt.title("Anomaly Detection : Isolation Forest Method")
plt.legend()
plt.show()