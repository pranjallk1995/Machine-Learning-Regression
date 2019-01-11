import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
dataset = pd.read_csv("Position_Salaries.csv")
X = pd.DataFrame(dataset.iloc[:, 1])
Y = pd.DataFrame(dataset.iloc[:, 2])

#no need to add X_0 to X since X is one-dimensional.
#because the data is very less, no need to split data, so as to improve the accuracy of training as much as possible.

#feature scaling is a must in SVR and SVM... why? see notes SVM section.
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
Y_sc = StandardScaler()
X = X.astype('float')           #converting datatypes from int to float (scaled values are float).
Y = Y.astype('float')
X = X_sc.fit_transform(X)
Y = Y_sc.fit_transform(Y)

#fitting SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', C = 10)         #Radical Basis Function kernel, same as Gaussian kernel.
#C default = 1, but here taking C large helps since it includes the outlier level 10 (CEO) salary.
#But if C is too large, then accuracy of prediction suffers... why? see notes.
regressor.fit(X, Y.ravel())                     # .ravel() is used to avoid warning. Y was a vector, but SVR package required an array.

#visualizing SVR
X_big = pd.DataFrame(np.arange(min(X), max(X), 0.1))
plt.scatter(X, Y, color = "red")
plt.plot(X_big, regressor.predict(X_big), color = "blue")             #important
plt.title("Position Level vs Salary (SVR model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#predicting value
y_pred = Y_sc.inverse_transform(regressor.predict(X_sc.transform(pd.DataFrame([6.5]))))
print(y_pred)