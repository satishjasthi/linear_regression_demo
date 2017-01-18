import pandas as pd;import numpy as np
import matplotlib.pyplot as plt;import seaborn as sns
from sklearn import linear_model,cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#read data
dataframe = pd.read_csv('challenge_dataset.txt',names = ['Brain','Body'])
features = dataframe
poly = PolynomialFeatures(2)
features = poly.fit_transform(dataframe)
features = pd.DataFrame(features,columns =['constant','Brain','Body','Brain**2','Brain*Body','Body**2'])
target = features['Body']
features = features.drop('Body',axis= 1);

#training classifier
lr = LinearRegression()
predicted = cross_val_predict(lr,features,target,cv = 5)
scores = cross_val_score(lr,features,target,cv=5,scoring = 'r2') # r2 values over different folds

#using k-fold cross_validation to split train and test classifier
loo = cross_validation.LeaveOneOut(len(target))
regr = LinearRegression()
scores = cross_validation.cross_val_score(regr, features, target, scoring='mean_squared_error', cv=loo,)
print "Mean squared error :",scores.mean()*(-1) # average Mean squared error : 3.07213345844
print "Root Mean squared error :",np.sqrt(scores.mean()*(-1)) #Root Mean squared error : 1.75275025558
