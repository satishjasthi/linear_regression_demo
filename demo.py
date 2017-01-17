import pandas as pd;import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import cross_val_score

#read data
dataframe = pd.read_csv('challenge_dataset.txt',names = ['Brain','Body'])
features = dataframe
poly = PolynomialFeatures(2)
features = poly.fit_transform(dataframe)
features = pd.DataFrame(features,columns =['constant','Brain','Body','Brain**2','Brain*Body','Body**2'])
target = features['Body']
features = features.drop('Body',axis= 1);

lr = LinearRegression()
predicted = cross_val_predict(lr,features,target,cv = 5)
scores = cross_val_score(lr,features,target,cv=5,scoring = 'r2')
scores


fig,ax = plt.subplots()
ax.scatter(target,predicted)
ax.plot([target.min(),target.max()],[target.min(),target.max()],'k--',lw=4)
ax.set_xlabel("Features")
ax.set_ylabel("Body weight")
plt.show()
