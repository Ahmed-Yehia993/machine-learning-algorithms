# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 16:38:19 2018

@author: Ahmed Yehia
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train,  regressor.predict(X_train), color='green')
plt.title('Salary vs Experince (Training set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train,  regressor.predict(X_train), color='green')
plt.title('Salary vs Experince (Training set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show()