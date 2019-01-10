#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoerting the dataset
dataset = pd.read_csv("Data.csv")
X = pd.DataFrame(dataset.iloc[:, :-1].values)   # [:, :-1] means all rows, all columns except the last column.
Y = pd.DataFrame(dataset.iloc[:, 3].values)     # [:, 3] means all rows of the 4th column.

#print(X.head())
#print(Y.head())
#print(dataset.head())