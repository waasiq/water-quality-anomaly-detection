"""
This module parses the data and returns X and Y data. 
We are parsing the data from Water Data
"""
import pandas as pd


def readFile():
    dataset = pd.read_csv('../Data/Water_Data.csv')

    # Convert the string to pandas datetime
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) 
    
    # Filling the missing NaN data in Missing data pad = propagate last valid observation to next value
    dataset = dataset.fillna(method = 'pad')

    
    X = dataset.iloc[:,2:3].values # other variables - to send to model
    y = dataset.iloc[:,0].values  # time
    
    return dataset,X,y
