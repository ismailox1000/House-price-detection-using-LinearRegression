# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 22:20:28 2022

@author: ismail_oubah
"""
#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#read csv file

dataset=pd.read_csv('houses.csv')
dataset.head()
dataset=dataset.drop(['waterfront','view'],axis=1)

#replace the missing data if any 
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(dataset)
dataset=imp.transform(dataset)
#split the data to work on training and testing later
X=dataset[:,:-1]
X.shape
y=dataset[:,-1]
y.shape
#scaling the data to be more faster and easy to work on

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X=std.fit_transform(X)

#now we split the data to training part and testing 80% training and 20% for testing

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
X_test
y_train
y_test
X_train.shape
y_train.shape
#predict the value of a variable based on the value of training data 
from sklearn.linear_model import LinearRegression
Regressor=LinearRegression( fit_intercept=True,normalize=True,n_jobs=-1)
Regressor.fit(X_train, y_train)
#scoring
Regressor.score(X_train, y_train)
Regressor.score(X_test,y_test)
#predict values based on X_test
y_pred=Regressor.predict(X_test)
y_test
y_pred
print('Linear Regression Train Score is : ' , Regressor.score(X_train, y_train))
print('Linear Regression Test Score is : ' , Regressor.score(X_test, y_test))
print('Linear Regression Coef is : ' , Regressor.coef_)
print('Linear Regression intercept is : ' , Regressor.intercept_)

plt.scatter(y_test, y_pred, color = 'blue')
plt.show()

#showing errors 
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error Value is : ', MAE)
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test, y_pred)
print('Mean Squared Error Value is : ', MSE)
from sklearn.metrics import median_absolute_error
MDSE=median_absolute_error(y_test, y_pred)
print('Median Squared Error Value is : ', MDSE )



'''

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_train2 = poly_reg.fit_transform(X_train)
X_test2 = poly_reg.fit_transform(X_test)
X_train2.shape 
X_test2.shape 


Regressor.fit(X_train2, y_train )

y_pred2 = Regressor.predict(X_test2) 

mean_absolute_error(y_test, y_pred2)


mean_squared_error(y_test, y_pred2)

median_absolute_error(y_test, y_pred2)

'''