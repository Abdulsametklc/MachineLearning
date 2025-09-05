import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("odev_tenis.txt")

from sklearn import preprocessing 

hava_durumu = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
hava_durumu[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
hava_durumu = ohe.fit_transform(hava_durumu).toarray() #hava durumunu 0-1 değerlerine çevirdik

ruzgar = veriler.iloc[:,3:4].values
le2 = preprocessing.LabelEncoder()
ruzgar[:,0] = le2.fit_transform(veriler.iloc[:,3])
ohe2 = preprocessing.OneHotEncoder()
ruzgar = ohe2.fit_transform(ruzgar).toarray()# windy sutununudaki değerleri 0-1 değerlerine çevirdik

play = veriler.iloc[:,-1:].values
le1 = preprocessing.LabelEncoder()
play[:,-1] = le1.fit_transform(veriler.iloc[:,-1])
ohe1 = preprocessing.OneHotEncoder()
play = ohe1.fit_transform(play).toarray() # -> oynama durumunu yes-no ikilisini sayısal değere, 0-1 değerlerine çevirdik

te_hu = veriler.iloc[:,1:3].values

sonuc = pd.DataFrame(data=hava_durumu, index=range(14), columns=['sunny','overcast','rainy'])
sonuc2 = pd.DataFrame(data=te_hu, index=range(14), columns=['temperatures','humidity'])
sonuc3 = pd.DataFrame(data=ruzgar, index = range(14), columns=['windy_false', 'windy_true'])
sonuc4 = pd.DataFrame(data=play, index=range(14), columns=['play_yes', 'play_no'])


s = pd.concat([sonuc, sonuc2, sonuc3], axis=1) 
s2 = pd.concat([s,sonuc4], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc4, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test) # -> 1. tenis oynanır mı oynanmaz mı play feature tahmini

#-------------------------------------------
temperature = veriler.iloc[:,1:2]
humidity = veriler.iloc[:,2:3]

s3= pd.concat([sonuc3, sonuc4, ], axis=1)
x2_train, x2_test, y2_train, y2_test = train_test_split(s3, humidity, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x2_train, y2_train)
y2_pred = regressor.predict(x2_test) # -> 2. humidity feature tahmini 

# y_test gerçek değerler
# y_pred tahmin edilen değerler
plt.scatter(y2_test, y2_pred, color="blue", alpha=0.6)
plt.xlabel("Gerçek Y değerleri")
plt.ylabel("Tahmin Y değerleri")
plt.title("Gerçek vs Tahmin")
plt.plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], "r--")  # referans doğru
plt.show()
#-----------------------------------
#Geri Eleme - Backward Elimination
import statsmodels.api as sm # -> İstatistiksel modelleme 

sag = s2.iloc[:,0:4]
sol = s2.iloc[:,4:8]
X_l = s2.iloc[:,[0,1,2,3,5,6,7,8]].values # alınan sütunlar
X_l= np.array(X_l, dtype=float) 
model = sm.OLS(humidity, X_l).fit() # humidity bağımlı değişken oluyor ve regresyon modeli eğitiliyor
print(model.summary())


X_la = s2.iloc[:,[5,6,7,8]].values # -> istatistikler sonucunda P=0.05 değerinden büyük değerlere sahip değişkenler elendi. p değeri 0 en yakın değerler tutuldu
X_la= np.array(X_la, dtype=float)
model = sm.OLS(humidity, X_la).fit()
print(model.summary())