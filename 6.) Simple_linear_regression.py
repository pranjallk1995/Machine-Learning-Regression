#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = pd.DataFrame(dataset.iloc[:, :-1].values)
header = ["Experience"]
X.columns = header
Y = pd.DataFrame(dataset.iloc[:, 1].values)
header = ["Salary"]
Y.columns = header

#splitting the data into training, cross-validation and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#print(X_test.head())
#print(X_train.head())
#print(Y_test.head())
#print(Y_train.head())

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results
Y_pred = pd.DataFrame(regressor.predict(X_test))
header = ["Predicted_Salary"]
Y_pred.columns = header

#visualizing training data
plt.scatter(X_train, Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#visualizing test data
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()