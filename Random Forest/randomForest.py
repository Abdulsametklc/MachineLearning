# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:06:01 2025

@author: Abdulsamet
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.txt')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values
Z = X + 0.5
K = X - 0.4


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color ='red')
plt.plot(X,rf_reg.predict(X), color ='blue')

plt.plot(X, rf_reg.predict(Z),color='green')
plt.plot(x, rf_reg.predict(K),color='yellow')