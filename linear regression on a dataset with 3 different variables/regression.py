import numpy as np;import pandas as pd;

#importing data
data = pd.read_fwf('autompgdataNaNRemoved.txt',names = ["mpg", "cylinders", "displacement", "horsepower","weight","acceleration","model_year","origin","car_name"])
