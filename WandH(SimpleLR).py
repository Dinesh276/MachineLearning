# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
dataset=pd.read_csv('WandH.csv')
print(dataset)
x=dataset.iloc[:,0:1]
y=dataset.iloc[:,1:0:-1]
print(x)
print(y)

#spliting training data and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#applying simple linear rigression
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(x_train,y_train)


#predicting the values using test
y_pred=slr.predict(x_test)
from sklearn.metrics import mean_squared_error
accuracy_regression = mean_squared_error(y_test,y_pred)
print(accuracy_regression)
acc= slr.score(y_test,y_pred)
print(acc)

#ploting
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,slr.predict(x_train),color="black")
plt.title("BMI")
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

#pridiction 
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,slr.predict(x_train),color="black")
plt.title("BMI")
plt.xlabel('height')
plt.ylabel('weight')
plt.show()