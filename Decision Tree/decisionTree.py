# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:39:30 2025

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

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X,Y, color = 'red')
plt.plot(X, r_dt.predict(X), color = 'blue')

plt.plot(x, r_dt.predict(Z), color = 'green')
plt.plot(x, r_dt.predict(K), color = 'yellow')

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

