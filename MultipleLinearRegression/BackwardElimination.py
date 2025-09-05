#Geri Eleme - Backward Elimination
import statsmodels.api as sm # -> İstatistiksel modelleme 

X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1) # bağımsız değişkenlerin başına 1 eklenmesi

X_l = veri.iloc[:,[0,1,2,3,4,5]].values # alınan sütunlar
X_l= np.array(X_l, dtype=float) 
model = sm.OLS(boy, X_l).fit() # boy bağımlı değişken oluyor ve regresyon modeli eğitiliyor
print(model.summary())


X_la = veri.iloc[:,[0,1,2,3,5]].values # -> istatistikler sonucunda P=0.05 değerinden büyük değerlere sahip değişkenler elendi. p değeri 0 en yakın değerler tutuldu
X_la= np.array(X_la, dtype=float)
model = sm.OLS(boy, X_la).fit()
print(model.summary())
