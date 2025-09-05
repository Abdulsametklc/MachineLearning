import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("odev_tenis.txt")

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform) 

c = veriler2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data=c, index=range(14), columns = ['overcast','rainy','sunny'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:], sonveriler],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,1:-1], sonveriler.iloc[:,-1:], test_size = 0.33, random_state=0) #sutun 1 p değeri 0.05'den yuksek çıktığı için eledik ve test-train verileri içine dahil etmedik

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1], axis=1)

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler = sonveriler.iloc[:,1:]

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())