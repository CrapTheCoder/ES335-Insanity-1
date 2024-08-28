from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tree import utils

np.random.seed(42)


@dataclass
class Node:
    feature: str = None
    threshold: float = None
    left: 'Node' = None
    right: 'Node' = None
    value: int = None


@dataclass
class DecisionTree:
    criterion: Literal["entropy", "gini_index", "mse"]
    is_real_input: bool
    is_real_output: bool
    max_depth: int = 5
    root: Node = None

    def __post_init__(self):
        """
        Check if the criterion is valid for the output type
        """
        if self.is_real_output and self.criterion != "mse":
            raise ValueError("Only MSE is supported for real output")

        if not self.is_real_output and self.criterion == "mse":
            raise ValueError("Only Information Gain or Gini Index is supported for discrete output")

    def get_leaf_value(self, y: pd.Series) -> int:
        """
        Get the leaf value based on the criterion.
        If the criterion is MSE, return the mean of the target variable.
        If the criterion is Information Gain or Gini Index, return the mode of the target variable.
        """
        return y.mean() if self.is_real_output else y.mode().values[0]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the decision tree model based on the input data
        Time complexity: O(N * M * D^2), where N is the number of samples and D is the maximum depth of the tree
        """

        def build(X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
            if depth >= 0:
                best_gain, best_feature, best_threshold = utils.optimal_split(X, y, self.criterion, X.columns)

                if best_gain > 0:
                    x_left, y_left, x_right, y_right = utils.split_data(X, y, best_feature, best_threshold)

                    left = build(x_left, y_left, depth - 1)
                    right = build(x_right, y_right, depth - 1)

                    return Node(best_feature, best_threshold, left, right)

            return Node(value=self.get_leaf_value(y))

        self.root = build(X, y, self.max_depth)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the target variable based on the fitted model for the input data
        Time complexity: O(N * D), where N is the number of samples and D is the maximum depth of the tree
        """
        def predict_row(node: Node, row: pd.Series) -> int:
            if node.value is not None:
                return node.value

            if row[node.feature] <= node.threshold:
                return predict_row(node.left, row)
            else:
                return predict_row(node.right, row)

        return X.apply(lambda row: predict_row(self.root, row), axis=1)

    def plot(self) -> None:
        """
        Print the decision tree structure in text format
        """
        left_symbol, right_symbol = ('<=', '> ') if self.is_real_input else ('==', '!=')

        def print_tree(node: Node, depth: int) -> None:
            if node is None:
                return

            line = "|   " * depth + "|---"
            if node.value is not None:
                print(line, "class:", node.value)
            else:
                print(line, f"{node.feature} {left_symbol} {node.threshold}")
                print_tree(node.left, depth + 1)
                print(line, f"{node.feature} {right_symbol} {node.threshold}")
                print_tree(node.right, depth + 1)

        print_tree(self.root, 0)
