import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pip as sns
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import streamlit as st

dosyaYolu = "C://Users//omer.dilaveroglu//Desktop//UiPathProjects//Python//KursDersleri//MachineLearning//data" 
dosyalar =os.listdir(dosyaYolu)

sutunlar=['model', 'year', 'price', 'transmission', 'mileage', 'fuelType','engineSize','marka']
veriler = {}
for dosya in dosyalar:
  dosya =dosyaYolu+"//"+dosya
  if dosya.endswith(".csv") and not os.path.basename(dosya).startswith("unclean"):
    df = pd.read_csv(dosya)
    df['marka'] = os.path.basename(dosya)[0:-4]
    df = df[sutunlar]
    veriler[dosya] = df
son = pd.concat(list(veriler.values()))

modeller=son['model'].unique()
markalar=son['marka'].unique()
ft=son['fuelType'].unique()
vitesler=son['transmission'].unique()
print(son['price'])


markasec=st.sidebar.selectbox("Marka",markalar)
modelsec=st.sidebar.selectbox("Model",modeller)
yakitsec=st.sidebar.selectbox("Yakıt Tipi",ft)
vitessec=st.sidebar.selectbox("Vites",vitesler)
yilsec=st.sidebar.number_input("Yıl",value=2015)
mileage=st.sidebar.number_input("Mil",value=20000)
motorsec=st.sidebar.number_input("Motor Hacmi" , min_value=0.5,value=1.6 ,max_value=8.0)

tahmin=son.copy()
tahmin=tahmin.iloc[[0]]
tahmin.iloc[0]=[modelsec,yilsec,0,vitessec,mileage,yakitsec,motorsec,markasec] #değiştir

son=pd.concat((son,tahmin),ignore_index=True)#ignore_index = sıradaki index numarasını verir
# fiyatı etkileyen herşey dami edilir. sağa doğru sutunlar ekleyerek 1 0 şeklinde sayısal değerlere çevrilir
#daha hızlı çalışmasını sağlar
#get_dummies = kategorisel veriyi sayısal veriye dönüştürür.
df=pd.get_dummies(son,columns=["model","transmission","fuelType","marka"],drop_first=True)#n tane cevabı olan bir soruyu n-1 tane soru ile bulabiliriz.


tahmin_array=df.iloc[[-1]]
tahmin_array=tahmin_array.drop("price",axis=1)

df=df.iloc[:-1]
y=df['price']
x=df.drop('price',axis=1)

hesapla=st.sidebar.button("Hesapla") # buton ekledim
if hesapla: #koşul ekledim diğerleri bir tab içeride
    lr=LinearRegression()
    model=lr.fit(x,y)
    skor=model.score(x,y)
    sonuc=model.predict(tahmin_array)
    st.title(sonuc) #yeni ekledim
    st.write("Başarı Oranı Skor:",skor) #yeni ekledim

