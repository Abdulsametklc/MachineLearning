import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri yükleme
veriler = pd.read_csv("maaslar.txt")

#data frame dilimleme
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

#lineer regresyon - doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y, color = 'red')
plt.plot(x, lin_reg.predict(X), color= 'blue')
plt.show()

#polinomal regresyon - doğrusal olmayan (nonlineer model) oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#görselleştirme
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color ='blue')
plt.show()

#4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)  #-> degree değeri arttırıldı, train verisi çok iyi okunur ve veri az ise yüksek değer girilmesi durumunda overfitting ihtimali artar. Model karmaşıklığı artar. 
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme
plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[7.5]]))

print(lin_reg3.predict(poly_reg3.fit_transform([[11]])))
print(lin_reg3.predict(poly_reg3.fit_transform([[7.5]])))                       