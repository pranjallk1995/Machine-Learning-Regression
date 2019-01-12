import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing data
dataset = pd.read_csv("Position_Salaries.csv")
X = pd.DataFrame(dataset.iloc[:, 1])
Y = pd.DataFrame(dataset.iloc[:, 2])

#no need to add X_0.
#no need to perform feature scaling. See logic of Decision Tree. (hint: it calculates standard deviation of all features.)

#fitting the Decision Tree.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

#visualizing Decision Tree.
#high resolution is required to see the means (prediction values) of different classes other wise its just poitns joined by straight lines.
X = X.values              #converting X to numpy.ndarray to later convert X to X_big.
X_big = pd.DataFrame(np.arange(min(X), max(X), 0.01))           #more precision of 0.01 to get straighter lines.
plt.scatter(X, Y, color = "red")
plt.plot(X_big, regressor.predict(X_big), color = "blue")
plt.title = "Position Levels vs Salary"
plt.xlabel = "Position Level"
plt.ylabel = "Salary"
plt.show()

#predicting values
y_pred = regressor.predict(pd.DataFrame([6.5]))
print(y_pred)