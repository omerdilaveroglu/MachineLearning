# -*- coding: utf-8 -*-
"""Arda Titanic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lwukVn1COV6VHl_Gx5zzIzPjfIp5XF-R
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

df=sns.load_dataset("titanic")
df.head(3)

df=df[["survived","pclass","age","sibsp","fare","parch","embarked","who"]]

df.head(3)

dfdum=pd.get_dummies(df,columns=["embarked","who"],drop_first=True)



dfdum.head(3)
dfdum=dfdum.dropna()

y=dfdum['survived']
x=dfdum.drop("survived",axis=1)

rf=RandomForestClassifier()
model=rf.fit(x,y)
model.score(x,y)

dfdum.head(3)

result =  model.predict([[3,15,0,4.5,0,0,1,0,0]])

print(result)

