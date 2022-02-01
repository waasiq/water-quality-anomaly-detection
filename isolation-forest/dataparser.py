"""
This module parses the data and returns X and Y data. 
We are parsing the data from Water Data
"""
import pandas as pd


def readFile():
    dataset = pd.read_csv('../data/water-data-reduced.csv')

    # Convert the string to pandas datetime
    #dataset['timestamp'] = pd.to_datetime(dataset['timestamp']) 
    
    # Filling the missing NaN data in Missing data pad = propagate last valid observation to next value
    dataset = dataset.fillna(method = 'pad')

    
    X = dataset.iloc[:,1:4].values # other variables - to send to model
    y = dataset.iloc[:,:1].values  # dates
    
    return dataset,X,y
