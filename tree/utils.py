import numpy as np
import pandas as pd
from scipy.special import xlogy


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding of the categorical features
    """
    return pd.get_dummies(X).astype('float64')


def entropy(y: pd.Series) -> float:
    """
    Entropy calculation
    """
    probabilities = y.value_counts(normalize=True)
    entropy_value = -np.sum(xlogy(probabilities, probabilities))
    return entropy_value


def gini_index(y: pd.Series) -> float:
    """
    Gini index calculation
    """
    probabilities = y.value_counts(normalize=True)
    gini_value = 1 - np.sum(probabilities ** 2)
    return gini_value


def mse(y: pd.Series) -> float:
    """
    Mean squared error calculation
    """
    return np.mean((y - y.mean()) ** 2)


def information_gain(y: pd.Series, left: pd.Series, right: pd.Series, criterion: str) -> float:
    """
    Information gain calculation
    """
    criterion_func = None
    match criterion:
        case 'entropy':
            criterion_func = entropy
        case 'gini_index':
            criterion_func = gini_index
        case 'mse':
            criterion_func = mse

    parent_value = criterion_func(y)
    weighted_children = (len(left) / len(y)) * criterion_func(left) + (len(right) / len(y)) * criterion_func(right)
    
    return parent_value - weighted_children


def optimal_split(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Find the optimal split for the data by maximizing information gain
    """
    best_split = (-np.inf, None, None)

    for feature in features:
        for threshold in X[feature].unique():
            x_left, y_left, x_right, y_right = split_data(X, y, feature, threshold)
            if len(x_left) and len(x_right):
                if (current_gain := information_gain(y, y_left, y_right, criterion)) > best_split[0]:
                    best_split = (current_gain, feature, threshold)
    
    return best_split


def split_data(X: pd.DataFrame, y: pd.Series, feature, threshold):
    """
    Split the data based on the feature and threshold
    """
    x_left, y_left = X[X[feature] <= threshold], y[X[feature] <= threshold]
    x_right, y_right = X[X[feature] > threshold], y[X[feature] > threshold]
    return x_left, y_left, x_right, y_right


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    """
    Split the data into training and testing sets
    """
    position = int(len(X) * (1 - test_size))
    return X[:position], X[position:], y[:position], y[position:]
