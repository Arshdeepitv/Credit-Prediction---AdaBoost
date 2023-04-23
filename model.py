import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('credit_customers.csv')
x = df.drop('class', axis=1)
y = df['class']
enc = LabelEncoder()
y = enc.fit_transform(y)
xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8,
                                              stratify=y)
cols = xtrain.columns[xtrain.dtypes=='object']
encoders = []
for i in cols:
    enc = LabelEncoder()
    xtrain[i] = enc.fit_transform(xtrain[i])
    encoders.append(enc)

def process(data):
    i = 0
    for col in data:
        if data[col].dtypes=='object':
            enc = encoders[i]
            data[col] = enc.transform(data[col])
            i +=1
    return data
