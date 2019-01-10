import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data.
dataset = pd.read_csv("Position_Salaries.csv")
X = pd.DataFrame(dataset.iloc[:, 1])
Y = pd.DataFrame(dataset.iloc[:, 2])

#no need to add X_0 to X since it will be handled by the linear regression package.
#because the data is very less, no need to split data, so as to improve the accuracy of training as much as possible.

#fitting linear hypothesis
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

#linear hypothesis visualization
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color = "red")
plt.plot(X, regressor.predict(X), color = "blue")
plt.title("Position Level vs Salary (Linear model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")

#fitting polynomial hyopthesis
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = pd.DataFrame(poly_regressor.fit_transform(X))
#print(X_poly)
#applying linear regression to the modified features.
regressor2 = LinearRegression()
regressor2.fit(X_poly, Y)

#polynomial hypothesis visualization
plt.subplot(1, 2, 2)
#increasing the positon levels to get smoother curves
X_big = pd.DataFrame(np.arange(0.0, 10.0, 0.1))
plt.scatter(X, Y, color = "red")
plt.plot(X_big, regressor2.predict(poly_regressor.fit_transform(X_big)), color = "blue")             #important
plt.title("Position Level vs Salary (Polynomial model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#prediction of salary at position level 6.5
print(regressor.predict(pd.DataFrame([6.5])))                                   #linear model
print(regressor2.predict(poly_regressor.fit_transform(pd.DataFrame([6.5]))))    #polynomial model