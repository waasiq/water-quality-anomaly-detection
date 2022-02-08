# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:50:49 2022

@author: waasiq
"""

# Libraries
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import pandas as pd
import numpy as np


dataset = pd.read_csv('../data/water-data-all.csv')
dataset = dataset.fillna(method = 'pad')

# Convert the string to pandas datetime
#dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) 
    
# Filling the missing NaN data in Missing data pad = propagate last valid observation to next value
#dataset = dataset.fillna(method = 'pad')

    
X = dataset.iloc[:,1:4].values # other variables - to send to model
y = dataset.iloc[:,:1].values  # dates


#--------------------------- Start of test data split ----------------------

# Splitting the dataset into the training set and data set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#--------------------------- End of Test Data Split -----------------------

# Fitting the LOF model to the dataset
from sklearn.neighbors import LocalOutlierFactor
model = LocalOutlierFactor(n_neighbors = 20, novelty = True, contamination = 'auto')
model.fit(X_train)


# Predicting the result with Local Outlier  Method
y_pred = model.predict(X_test)
anomaly_score  = model.decision_function(X_test)

outcome = pd.DataFrame(data = y_test, index = range(1035), columns = ['timestamp'] )
outcomeAnamoly = pd.DataFrame(data = anomaly_score, index = range(1035), columns = ['anomaly_score'] )
inputData = pd.DataFrame(data = X_test, index = range(1035), columns = ['SC(uS)','Turb(FNU)','DO(mg/l)'])

outputFrame = pd.concat([outcomeAnamoly , inputData], axis = 1)

outputFrame.to_excel('../data/data-w-anomaly.xlsx')

outcomeConcat = pd.concat([outcome , outcomeAnamoly], axis = 1)

'''

# ----------------------- Graph Visualization Below -----------------------

# Visualizing the Isolation Forest Results anamoly w.r.t to Time
anomaly_data = outcomeConcat.loc[outcomeConcat['y_pred'] == -1, ['timestamp','y_pred']]

# Prints the percentage anomalies in data
print("Accuracy:", (list(y_pred).count(1)/y_pred.shape[0]) * 100)
print("Percentage of anomalies in data: {:.2f}".format((len(outcomeConcat.loc[outcomeConcat['y_pred']==-1])/len(outcomeConcat))*100))

plt.style.use('seaborn')
plt.figure(figsize=(12,8))

plt.plot(outcomeConcat['timestamp'],outcomeConcat['y_pred'], color='blue', label = 'Normal')
plt.plot_date(x=anomaly_data['timestamp'],y=anomaly_data['y_pred'], color='red', label = 'Anomaly')
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.xlabel("Datatime Index")
plt.ylabel("Anomaly")
plt.title("Anomaly Detection : Isolation Forest Method")
plt.legend()
plt.show()
'''