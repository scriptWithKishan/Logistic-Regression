import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
from model import Logistic_Regression

data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
y =data['Outcome']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state=42)

model = Logistic_Regression(lr=0.1, n_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)