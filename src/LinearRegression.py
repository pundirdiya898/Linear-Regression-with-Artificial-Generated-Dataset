# IMPORT LIBRARY
import pandas as pd
import numpy as np

# GENERATE DATASETS
from sklearn.datasets import make_regression
# without coefficient of underline model
X,y = make_regression(n_samples=500,n_features=5,coef=False,bias=12,noise=10,random_state=2529)
# with coefficient of underline model
X,y,w = make_regression(n_samples=500,n_features=5,coef=True,bias=12,noise=10,random_state=2529)
X.shape,y.shape
w

# Get first 5 rows of target vaiable and features
X[0:5]
y[0:5]

#GET SHAPE OF DATAFRAME
X.shape, y.shape

# GET TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# GET LINEAR REGRESSION MODEL TRAIN
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

#GET Intercept and cofficients
model.intercept_
model.coef_

# GET MODEL PREDICTION
y_pred = model.predict(X_test)
y_pred.shape
y_pred

# GET MODEL EVALUATION
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
mean_squared_error(y_test,y_pred)
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
r2_score(y_test,y_pred)
