import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from tree import base, utils
import metrics


def decision_tree_usage(X, y):
    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.3)

    for criteria in ['entropy', 'gini_index']:
        tree = base.DecisionTree(criterion=criteria, max_depth=5, is_real_input=True, is_real_output=False)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)

        tree.plot()

        print(f"Criteria used:", criteria)
        print()

        print("Accuracy:", metrics.accuracy(y_pred, y_test))
        for cls in y.unique():
            print("Precision:", metrics.precision(y_pred, y_test, cls))
            print("Recall:", metrics.recall(y_pred, y_test, cls))

        print('-' * 50)


def k_fold_split(X, y, k=5):
    indices = np.arange(X.shape[0])
    fold_size = X.shape[0] // k

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size

        test_indices = indices[start: end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        yield X_train, X_test, y_train, y_test


def nested_cross_validation(X, y, k=5, max_depth=10):
    outer_folds = k_fold_split(X, y, 5)
    outer_scores = []

    for outer_fold in outer_folds:
        X_outer_train, X_outer_test, y_outer_train, y_outer_test = outer_fold
        inner_folds = list(k_fold_split(X_outer_train, y_outer_train, k))

        best_depth, best_accuracy = None, -np.inf
        for depth in range(1, max_depth + 1):
            accuracy = 0
            for inner_fold in inner_folds:
                X_inner_train, X_inner_val, y_inner_train, y_inner_val = inner_fold

                tree = base.DecisionTree(criterion='entropy', max_depth=depth, is_real_input=True, is_real_output=False)
                tree.fit(X_inner_train, y_inner_train)

                y_pred = tree.predict(X_inner_val)
                accuracy += metrics.accuracy(y_pred, y_inner_val)

            accuracy /= k

            if accuracy > best_accuracy:
                best_depth, best_accuracy = depth, accuracy

        tree = base.DecisionTree(criterion='entropy', max_depth=best_depth, is_real_input=True, is_real_output=False)
        tree.fit(X_outer_train, y_outer_train)

        y_pred = tree.predict(X_outer_test)
        accuracy = metrics.accuracy(y_pred, y_outer_test)
        outer_scores.append((best_depth, accuracy))

        print("-> best depth, accuracy:", best_depth, accuracy)


    depths, accuracies = zip(*outer_scores)
    print("Average best depth:", np.mean(depths))
    print("Average accuracy:", np.mean(accuracies))


def main():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=2, class_sep=0.5)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    X = pd.DataFrame(X)
    y = pd.Series(y).astype('category')

    decision_tree_usage(X, y)
    nested_cross_validation(X, y, max_depth=10)


if __name__ == '__main__':
    main()
