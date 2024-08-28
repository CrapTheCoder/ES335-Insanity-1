import numpy as np
import pandas as pd

import metrics
from tree import base

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Loading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, sep='\s+', header=None, names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"])

# Removal of rows with missing values and unnecessary columns
data = data[data["horsepower"] != "?"]
data["horsepower"] = data["horsepower"].astype(float)

X = data.drop(["mpg", "car name"], axis=1)
y = data["mpg"]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# Our Decision Tree implementation
my_model = base.DecisionTree(criterion="mse", max_depth=20, is_real_input=True, is_real_output=True)
my_model.fit(X_train, y_train)
y_pred = my_model.predict(X_test)

print('RMSE of Decision Tree implementation:', metrics.rmse(y_pred, y_test))

# Scikit Decision Tree
scikit_model = DecisionTreeRegressor(max_depth=20)
scikit_model.fit(X_train, y_train)
y_pred = scikit_model.predict(X_test)

print('RMSE of Scikit Decision Tree:', metrics.rmse(y_pred, y_test))
