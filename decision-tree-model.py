#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:57:10 2018

@author: alex
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Import dataset
dataset_file_path = './melb_data.csv'
data = pd.read_csv(dataset_file_path)

data = data.dropna(axis=0)
price = data.Price

predictors_columns = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Bedroom2']

data_columns = data[predictors_columns]

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(data_columns, price,random_state = 0)
# Define model
data_model = DecisionTreeRegressor()
#Fit model
data_model.fit(train_X, train_y)

val_predictions = data_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


