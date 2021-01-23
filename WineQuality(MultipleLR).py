# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 11:58:12 2021

@author: D1NU
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


#importing data set
data=pd.read_csv("WineQuality.csv")
x=data.iloc[:,:11].values
y=data.iloc[:,11:12].values

sea.heatmap(data.corr(), annot=True)

#seperating test and train datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2,random_state=0)

#applying regression 
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)

#predicting values from model
y_pred=np.round(mlr.predict(x_test))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#error calculation
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error: ', MSE)
print('R2 Score: ', r2)


#optimizing data
import statsmodels.api as sm
x=np.append(arr=np.ones((1599,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt=x[:,[0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt=x[:,[0,1,2,3,5,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt=x[:,[0,1,2,5,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())
x_opt=x[:,[0,2,5,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
print(regressor_OLS.summary())

#splitting test and train data from optimized data 
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1= train_test_split(x_opt,y,test_size=.2,random_state=0)

#new model based on optimized data
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train1,y_train1)

#predicting values
y_pred1=np.round(mlr.predict(x_test1))

#calculating data
from sklearn.metrics import mean_squared_error, r2_score
MSE = mean_squared_error(y_test1, y_pred1)
r2 = r2_score(y_test1, y_pred1)

from sklearn.metrics import classification_report
print(classification_report(y_test1, y_pred1))

print('Mean squared error: ', MSE)
print('R2 Score: ', r2)

