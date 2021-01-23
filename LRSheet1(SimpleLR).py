# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:46:51 2021

@author: D1NU
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data set
dataset=pd.read_csv('LRSheet1.csv')
x=dataset.iloc[:,0:1]
y=dataset.iloc[:,1:2]


plt.scatter(x,y,color="red",marker="*")
plt.plot(x,y,color='black')
plt.title('Simple linear regression')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()


#ignoring the last to data as they seem to not lie on the line and may effect the performance of the model.

x=dataset.iloc[:298,0:1]
y=dataset.iloc[:298,1:2]

#splitting dataset
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=.2,random_state=0)


#fitting data to linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xTrain,yTrain)

#predicting values and error
yPred=regressor.predict(xTest)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, xTrain, yTrain, cv= 5)
print("scores :",scores)


from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(yTest, yPred)
r2 = r2_score(yTest, yPred)

print('Mean squared error: ', MSE)
print('R2 Score: ', r2)



#ploting training data
plt.scatter(xTrain,yTrain,color="red",marker="*")
plt.plot(xTrain,regressor.predict(xTrain),color='black')
plt.title('Simple linear regression on train data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()



#plotting test datas
plt.scatter(xTest,yTest,color="red",marker="*")
plt.plot(xTrain,regressor.predict(xTrain),color='black')
plt.title('Simple linear regression on test data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()