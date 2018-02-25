#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:57:10 2018

@author: alex
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn as sk
# Import dataset
dataset_file_path = './melb_data.csv'
data = pd.read_csv(dataset_file_path)

data = data.dropna(axis=0)
price = data.Price

predictors_columns = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Bedroom2']

data_columns = data[predictors_columns]

data_model = DecisionTreeRegressor()

data_model.fit(data_columns, price)

print("Predictions of prices on", len(price), "houses")
print("Accuracy: ",sk.metrics.accuracy_score(price, data_model.predict(data_columns)) ,"%")



