#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")
X = pd.DataFrame(dataset.iloc[:, :-1].values)
header = ["Country", "Age", "Salary"]
X.columns = header
Y = pd.DataFrame(dataset.iloc[:, 3].values)
header = ["Purchased"]
Y.columns = header

#taking care of missing data
from sklearn import impute
imputer = impute.SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
classifier_encoder = LabelEncoder()
Y["Purchased"] = classifier_encoder.fit_transform(Y["Purchased"])

X = pd.get_dummies(X, columns=['Country'], prefix = ['Country'])

#splitting the data into training, cross-validation and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#print(X_train.head())