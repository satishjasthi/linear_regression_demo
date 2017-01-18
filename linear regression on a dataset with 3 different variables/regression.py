import numpy as np;import pandas as pd;
import matplotlib.pyplot as plt; import seaborn as sns;
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#importing data
dataframe = pd.read_fwf('autompgdataNaNRemoved.txt',names = ["mpg", "cylinders", "displacement", "horsepower","weight","acceleration","model_year","origin","car_name"])
data = dataframe.drop('cylinders',axis=1) #remove cylinders as it is multi-valued discrete

#visualize the relationship between variables
#x = displacement y = mpg
plt.scatter(data.displacement,data.mpg)
plt.xlabel("displacement");plt.ylabel("mpg");plt.show()

#taking log transformation of displacement and mpg features to know whether they can show a linear relationship
#x = displacement y = mpg
logvar = ['mpg','displacement']
logdata = pd.DataFrame().reindex_like(data[['mpg','displacement']])#create a dummy dataframe of required size
logdata[logvar] = data[logvar].apply(lambda x: np.log(x+1))
plt.scatter(logdata.displacement,logdata.mpg)
plt.xlabel("displacement");plt.ylabel("mpg");plt.show()

#x = horsepower y = mpg
plt.scatter(data.horsepower,data.mpg)
plt.xlabel("horsepower");plt.ylabel("mpg");plt.show()

#x = weight y = mpg
plt.scatter(data.weight,data.mpg)
plt.xlabel("weight");plt.ylabel("mpg");plt.show()

#x = acceleration y = mpg
plt.scatter(data.acceleration,data.mpg)
plt.xlabel("acceleration");plt.ylabel("mpg");plt.show()

#defining features and targets
features = data[['displacement','horsepower','weight','acceleration']]
targets = data['mpg']


#train_test_split data
features_train,features_test,targets_train,targets_test = train_test_split(features,targets,test_size=0.25,random_state=42)

#training classifier
reg = LinearRegression()
reg.fit(features_train,targets_train)
pred = reg.predict(features_test)
print "Mean squared error :",mean_squared_error(targets_test,pred)
