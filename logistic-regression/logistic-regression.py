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
dataset = pd.read_csv('..\Doc\Water_Data.csv')

dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) 

time = dataset.iloc[:,0:1]
#------------------------------------------
# Missing values ​​are taken as the mean value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
missingValues = dataset.iloc[:, 1:4].values

imputer = imputer.fit((missingValues[:,0:3]))
missingValues[:,0:3] = imputer.transform(missingValues[:,0:3])
print(missingValues)

#------------------------------------------
outcome = pd.DataFrame(data=missingValues, index=range(5171), columns=["SC(uS)", "Turb(FNU)", "DO(mg/L)"])
print(outcome)

outcomeConcat = pd.concat([time, outcome], axis = 1)
print(outcomeConcat)

#------------------------------------------
x = outcomeConcat.iloc[:,1:]
y = outcomeConcat.iloc[:,0:1]

X = x.values # the other variables
Y = y.values # dates

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X,Y)

Y_pred = logr.predict(X) #Y_pred is a date

print(Y_pred)
anomaly_score  = logr.decision_function(X)
#quality = pd.DataFrame(data=Y_pred, index=range(99), columns=["Quality"])

# quality ekrana yazdır Plot bak
# ----------------------- Graph Visualization Below -----------------------
# Visualizing the Logistic Regression Results
outcomeConcat['anomaly']= pd.Series(Y_pred)
#anomaly_data = outcomeConcat.loc[outcomeConcat['anomaly'] == -1, ['timestamp','Turb(FNU)','SC(uS)','DO(mg/L)']]

plt.style.use('seaborn')
plt.figure(figsize=(12,8)) # Set the figure size


plt.plot(outcomeConcat['timestamp'],outcomeConcat['anomaly'], color='blue', label = 'Normal')
#plt.plot(outcomeConcat['timestamp'],outcomeConcat['SC(uS)'], color='green', label = 'Normal')
#plt.plot(outcomeConcat['timestamp'],outcomeConcat['DO(mg/L)'], color='black', label = 'Normal')
#plt.scatter(outcomeConcat['timestamp'],outcomeConcat['Turb(FNU)'],color='red', label = 'Anomaly Turb(FNU)')
plt.scatter(outcomeConcat['timestamp'],outcomeConcat['anomaly'],color='black', label = 'Anomaly DO(mg/L)')
#plt.scatter(outcomeConcat['timestamp'],outcomeConcat['SC(uS)'],color='green', label = 'Anomaly SC(uS)')

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d, %b, %Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.xlabel("Datatime Index")
plt.ylabel("Water Quality")
plt.title("Anomaly Detection : Logistic Regression Method")
plt.legend()
plt.show()




