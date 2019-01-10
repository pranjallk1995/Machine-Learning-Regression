import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('50_Startups.csv');
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 4]

#adding 1's at X_0
X.insert(0, "X_0", 1)

#encoding data
X = pd.get_dummies(X, columns=['State'], prefix = ['State'])

#avoiding dummy variable trap. (already done by the package linear_model)
#X = X.iloc[:, :-1]
#print(X)

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling is done by the package linear_model

#multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#check predictions
Y_pred = pd.DataFrame(regressor.predict(X_test))

#print(Y_pred)
#print(Y_test)

#building the optimal model using backward elimination.
import statsmodels.formula.api as sm
#X_opt = X.iloc[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

#max p-value for X_4 > 5%, Hence we remove X_4
#X_opt = X.iloc[:, [0, 1, 2, 3, 5]]
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

#max p-value for X_5 > 5%, Hence we remove X_5
#X_opt = X.iloc[:, [0, 1, 2, 3]]
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

#max p-value for X_2 > 5%, Hence we remove X_2
#X_opt = X.iloc[:, [0, 1, 3]]
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

#max p-value for X_3 > 5%, Hence we remove X_3
#X_opt = X.iloc[:, [0, 1]]
#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

#OR

def backwardElimination(x, sl):
    numVars = len(x.columns)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = Y, exog = x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    x = x.drop(x.columns[j], axis = 1)
    print(regressor_OLS.summary())
    return x


SL = 0.05
X_opt = X.iloc[:, [0, 1, 2, 3, 4, 5]]               #This step is done to name the columns 0, 1, ... to X_opt for easier deletion etc.
X_Modeled = backwardElimination(X_opt, SL)          #Can't send X directly.

#comparing results from optimal model (dimensionally reduced)
X_Modeled_train, X_Modeled_test, Y_Modeled_train, Y_Modeled_test = train_test_split(X_Modeled, Y, test_size = 0.2, random_state = 0)
regressor.fit(X_Modeled_train, Y_Modeled_train)
Y_Modeled_pred = pd.DataFrame(regressor.predict(X_Modeled_test))

#visualization of comparision
plt.subplot(1, 2, 1)
plt.scatter(X_test["R&D Spend"], Y_pred, color = "blue")
plt.title("Profit vs R&D Spend using all the features")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.subplot(1, 2, 2)
plt.scatter(X_Modeled_test["R&D Spend"], Y_Modeled_pred, color = "blue")
plt.title("Profit vs R&D Spend after feature reduction")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.show()

#Hence its very similar!