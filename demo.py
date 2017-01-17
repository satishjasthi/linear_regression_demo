import pandas as pd;import numpy as np
import matplotlib.pyplot as plt;import seaborn as sns
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score,train_test_split
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#read data
dataframe = pd.read_csv('challenge_dataset.txt',names = ['Brain','Body'])
features = dataframe
poly = PolynomialFeatures(2)
features = poly.fit_transform(dataframe)
features = pd.DataFrame(features,columns =['constant','Brain','Body','Brain**2','Brain*Body','Body**2'])
target = features['Body']
features
sns.pairplot(features,x_vars = ['Brain','Brain**2','Brain*Body','Body**2'],y_vars = 'Body',kind = 'reg')
features = features.drop('Body',axis= 1)

#splitting data into training and testing data
features_train,features_test,target_train,target_test = train_test_split(features,target,test_size=0.25,random_state = 0)

#training classifier
lr = LinearRegression()
lr.fit(features_train,target_train)
print "intercept of LR line:",lr.intercept_
print "co-efficients of LR line:", lr.coef_
pred = lr.predict(features_test)


predicted = cross_val_predict(lr,features,target,cv = 5)
scores = cross_val_score(lr,features,target,cv=5,scoring = 'r2')
scores
