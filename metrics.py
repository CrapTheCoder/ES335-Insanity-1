from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    assert y_hat.size == y.size

    return (y_hat == y).mean()


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size

    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()

    return tp / (tp + fp)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size

    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()

    return tp / (tp + fn)


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error (RMSE)
    """
    assert y_hat.size == y.size

    return ((y_hat - y) ** 2).mean() ** 0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error (MAE)
    """
    assert y_hat.size == y.size

    return (y_hat - y).abs().mean()
