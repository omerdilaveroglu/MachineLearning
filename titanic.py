import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df=sns.load_dataset("titanic")
print(df)
df=df[["survived","pclass","age","sibsp","fare","parch","embarked","who"]]
print(df)
dfdum=pd.get_dummies(df,columns=["embarked","who"],drop_first=True)
print(dfdum)

dfdum=dfdum.dropna()

y=dfdum['survived']
x=dfdum.drop("survived",axis=1)

rf=RandomForestClassifier()
model=rf.fit(x,y)
print(model.score(x,y))
print(model.predict([[3,15,0,4.5,0,0,1,0,0]]))

