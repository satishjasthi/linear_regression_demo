import numpy as np;import pandas as pd;
import matplotlib.pyplot as plt; import seaborn as sns;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#importing data
dataframe = pd.read_fwf('autompgdataNaNRemoved.txt',names = ["mpg", "cylinders", "displacement", "horsepower","weight","acceleration","model_year","origin","car_name"])
data = dataframe.drop('cylinders',axis=1) #remove cylinders as it is multi-valued discrete

#defining features and targets
features = data[['displacement','horsepower','weight','acceleration']]
targets = data['mpg'];mpg = targets

#visualize the relationship between variables
for var in ['displacement','horsepower','weight','acceleration']:
    sns.regplot(data[var],mpg)
    plt.show()

#train_test_split data
features_train,features_test,targets_train,targets_test = train_test_split(features,targets,test_size=0.25,random_state=42)

#training classifier
reg = LinearRegression()
reg.fit(features_train,targets_train)
pred = reg.predict(features_test)
print "Root mean squared error :",np.sqrt(mean_squared_error(targets_test,pred))

#train the model without displacement feature
features = features.drop('displacement',axis=1)

#train_test_split data
features_train,features_test,targets_train,targets_test = train_test_split(features,targets,test_size=0.25,random_state=42)

#training classifier
reg = LinearRegression()
reg.fit(features_train,targets_train)
pred = reg.predict(features_test)
print "Root mean squared error :",np.sqrt(mean_squared_error(targets_test,pred))

#train the model without acceleration feature
features = data[['displacement','horsepower','weight']];features

#train_test_split data
features_train,features_test,targets_train,targets_test = train_test_split(features,targets,test_size=0.25,random_state=42)

#training classifier
reg = LinearRegression()
reg.fit(features_train,targets_train)
pred = reg.predict(features_test)
print "Root mean squared error :",np.sqrt(mean_squared_error(targets_test,pred))
