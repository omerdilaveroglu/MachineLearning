import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sqlite3


df=pd.read_csv("sigorta.zip")
bolgeler=df['region'].unique()
aktif=0
with st.sidebar.form("ai"):
    isim=st.text_input("İsim Soyisim")
    telefon=st.text_input("Telefon")
    yas=st.number_input("Yaşınız")
    cinsiyet=st.selectbox("Cinsiyet",["male","female"])
    boyunuz=st.number_input("Boyunuz",min_value=1.0,max_value=2.5,value=1.69)
    kilonuz=st.number_input("Kilonuz",min_value=30.0,max_value=250.0,value=80.0)
    bmi=kilonuz/(boyunuz**2)
    cocuk=st.number_input("Çocuk Sayısı",step=1)

    sigara=st.selectbox("Sigara İçiyor Musun",["yes","no"])
    bolge=st.selectbox("Bölge",bolgeler)

    modelsec=st.selectbox("Model Seç",["Linear Regression","Random Forest"])
    train = st.slider("TrainSize", min_value=0.0, max_value=1.0, value=0.75)
    hesapla=st.form_submit_button("Hesapla")

    if hesapla:

        tahmin=df.iloc[[-1]].copy()
        tahmin.iloc[-1]=np.array([yas,cinsiyet,bmi,cocuk,sigara,bolge,0])
        df=pd.concat([df,tahmin])
        print(df.info())
        df=pd.get_dummies(df,columns=["sex","smoker","region"],drop_first=True)
        tahmin=df.iloc[[-1]].drop("charges",axis=1)
        df=df.iloc[0:-1]
        y=df['charges']
        x=df.drop("charges",axis=1)
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train,random_state=48)


        #LinearRegression
        lr=LinearRegression()
        model=lr.fit(x_train,y_train)
        skor_lr=model.score(x_test,y_test)

        #Random Forest
        rf=RandomForestRegressor()
        model=rf.fit(x_train,y_train)
        skor_rf=model.score(x_test,y_test)

        if modelsec=="Linear Regression":
            model=lr.fit(x_train,y_train)
        elif modelsec=="Random Forest":
            model=rf.fit(x_train,y_train)

        tahmin_rf=model.predict(tahmin)
        aktif=1

if aktif==1:
    st.title("Sonuç")
    st.write(tahmin_rf[0])
    st.subheader("Skor")
    st.write("Linear Regression:",skor_lr,"Random Forest:",skor_rf)

    conn=sqlite3.connect("ai.sqlite3")
    c=conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS kayitlar(isim TEXT,telefon TEXT,fiyat REAL)")
    conn.commit()
    c.execute("INSERT INTO kayitlar VALUES(?,?,?)", (isim, telefon,tahmin_rf[0]))
    conn.commit()
    st.success("Kayıt Başarılı")

goster=st.button("Geçmiş Kayıtları Göster")
if goster:
    conn = sqlite3.connect("ai.sqlite3")
    c = conn.cursor()
    c.execute("SELECT * FROM kayitlar")
    x=c.fetchall()
    st.dataframe(pd.DataFrame(x))

