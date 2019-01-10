#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")
X = pd.DataFrame(dataset.iloc[:, :-1].values)   # [:, :-1] means all rows, all columns except the last column.
header = ["Country", "Age", "Salary"]
X.columns = header
Y = pd.DataFrame(dataset.iloc[:, 3].values)     # [:, 3] means all rows of the 4th column.
header = ["Purchased"]
Y.columns = header

#taking care of missing data
from sklearn import impute
imputer = impute.SimpleImputer(missing_values = np.nan, strategy = "mean")       # all these parameter values are default, so no need to write.
imputer = imputer.fit(X.iloc[:, 1:3])            # (fit the corresponding imputer object onto imputer) Fit the column wise mean of all the values in the missing data in columns 1 and 2.
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

#print(X.head())
#print(Y.head())