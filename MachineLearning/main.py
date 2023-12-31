import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from xgboost import XGBClassifier

df = pd.read_excel("tmp001.xlsx")
df = df.dropna()

filter = df['TUTAR'] != 0
df = df[filter]

df.loc[len(df)] = ['ÖZEL OTOMOBİL',2011,'OPEL',
                            'CORSA 1.3 CDTi 5 KAPI ESSENTIA (01 / HUSUSİ OTOMOBİL)','5.BASAMAK',
                            'SÜRPRİMSİZ','SÜRPRİMSİZ',0,34]

df = pd.get_dummies(df,columns=['KULLANIM_TARZI','MARKA','TIP','BASAMAK',
                                'BASAMAK_SURPIRIM','GECIKMA_SURPIRIM'],drop_first=True)

tahmin = df.iloc[[-1]].drop('TUTAR',axis=1)
df=df.iloc[0:-1]
y=df['TUTAR']
x=df.drop('TUTAR',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70,random_state=60)

#LinearRegression
lr=LinearRegression()
model=lr.fit(x_train,y_train)
skor_lr=model.score(x_test,y_test)
tahmin_lr = model.predict(tahmin)
print(f"lr skor: {skor_lr} \n lr tahmin {tahmin_lr}")

#Random Forest
rf=RandomForestRegressor()
model=rf.fit(x_train,y_train)
skor_rf=model.score(x_test,y_test)
tahmin_rf = model.predict(tahmin)
print(f"lr skor: {skor_rf} \n lr tahmin {tahmin_rf}")

# ************************************************************
# aşağıdaki model öğrenmek için yüzde oranı belirtilmez bütün data öğrenmek için kullanılır. son data tahmin edilir

#LinearRegression
lr=LinearRegression()
model=lr.fit(x,y)
skor_lr=model.score(x,y)
tahmin_lr = model.predict(tahmin)
print(f"lr skor: {skor_lr} \n lr tahmin {tahmin_lr}")

#Random Forest
rf=RandomForestRegressor()
model=rf.fit(x,y)
skor_rf=model.score(x,y)
tahmin_rf = model.predict(tahmin)
print(f"lr skor: {skor_rf} \n lr tahmin {tahmin_rf}")

# XGB
"""
xgb =XGBClassifier()
model_xgb = xgb.fit(x_train,y_train)
xgb_skor = model_xgb.score(x_test,y_test)

print("XGB Skoru:",xgb_skor)
"""
