# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:37:11 2022

@author: waasiq
"""

# Libraries
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import pandas as pd

from dataparser import readFile

dataset,X,y = readFile()


# Taking  two sub data sets both for graph plotting and sub data
sub_dataset = dataset.filter(['timestamp','Turb(FNU)'], axis=1)
graph_plotter = sub_dataset



#--------------------------- Start of test data split ----------------------

'''
# Splitting the dataset into the training set and data set
# And preparing the frames for the test data  

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
X_test = X_test.astype('float64')

sub_dataset = pd.DataFrame(np.hstack((X_test,y_test))) #Convert to dataframe
# Renaming the columns from 0,1 to timestamp and placeholder
sub_dataset.columns = { 'Turb(FNU)' , 'placeholder'}
sub_dataset.rename(columns = {'placeholder' : 'timestamp'}, inplace = True)
sub_dataset['timestamp'] = pd.to_datetime(sub_dataset['timestamp'])
graph_plotter = sub_dataset 

'''
#--------------------------- End of Test Data Split -----------------------

# Fitting the IsolationForest model to the dataset
# If used for test model change the X to X_train in fitting 
# While cahnge the predict to the X_train and decision_function
from sklearn.ensemble import IsolationForest
model = IsolationForest(n_estimators = 100, max_samples = 'auto', contamination = 'auto', random_state = 42)
model.fit(X, y)

# Predicting the result with Isolation Forest Method
anamoly = model.predict(X)
anomaly_score  = model.decision_function(X)


# ----------------------- Graph Visualization Below -----------------------
# Cannot plot graph in mutliple dimensions so only the Time-Trubdity is being 
# plotted 

# Visualizing the Isolation Forest Results
graph_plotter['anomaly']= pd.Series(anamoly)
anomaly_data = graph_plotter.loc[graph_plotter['anomaly'] == -1, ['timestamp','Turb(FNU)']]

# Filling the NaN data in both the data frames
graph_plotter = graph_plotter.fillna(method = 'pad')
anomaly_data = anomaly_data.fillna(method = 'pad')


# Prints the percentage anomalies in data
print("Accuracy:", (list(anamoly).count(1)/anamoly.shape[0]) * 100)
print("Percentage of anomalies in data: {:.2f}".format((len(graph_plotter.loc[graph_plotter['anomaly']==-1])/len(graph_plotter))*100))

plt.style.use('seaborn')
plt.figure(figsize=(12,8)) # Set the figure size

plt.plot(graph_plotter['timestamp'],graph_plotter['Turb(FNU)'], color='blue', label = 'Normal')
plt.plot_date(x=anomaly_data['timestamp'],y=anomaly_data['Turb(FNU)'], color='red', label = 'Anomaly')

plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d, %b, %Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.xlabel("Datatime Index")
plt.ylabel("Turbidity Values(FNU)")
plt.title("Anomaly Detection : Isolation Forest Method")
plt.legend()
plt.show()