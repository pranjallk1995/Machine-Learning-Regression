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

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
classifier_encoder = LabelEncoder()
Y["Purchased"] = classifier_encoder.fit_transform(Y["Purchased"])

#print(Y.head())
#country_encoder = LabelEncoder()
#X.iloc[:, 0] = country_encoder.fit_transform(X.iloc[:, 0])
#print(X)       #This gives values like 0 to France, 1 to Spain etc. Giving a sense of rank, but there is no order in this feature.
X = pd.get_dummies(X, columns=['Country'], prefix = ['Country'])            # This method is called One-Hot encoding.

#print(X.head())