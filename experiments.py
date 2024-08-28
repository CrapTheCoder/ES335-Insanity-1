import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from tree import base, utils

np.random.seed(42)
num_average_time = 10


# Function to create fake data (take inspiration from usage.py)
def create_dataset(n, m, is_real_input, is_real_output):
    if is_real_input:
        x = pd.DataFrame(np.random.randn(n, m))
    else:
        x = pd.DataFrame({i: pd.Series(np.random.randint(2, size=n), dtype='category') for i in range(m)})

    if is_real_output:
        y = pd.Series(np.random.randn(n))
    else:
        y = pd.Series(np.random.randint(m, size=n), dtype='category')

    return x, y


def get_time(n, m, is_real_input, is_real_output, criteria, max_depth):
    x, y = create_dataset(n, m, is_real_input, is_real_output)
    tree = base.DecisionTree(criteria, is_real_input, is_real_output, max_depth)

    if not is_real_input:
        x = utils.one_hot_encoding(x)

    start_time = time.perf_counter()
    tree.fit(x, y)
    end_time = time.perf_counter()
    fit_time = end_time - start_time

    start_time = time.perf_counter()
    tree.predict(x)
    end_time = time.perf_counter()
    predict_time = end_time - start_time

    return fit_time, predict_time


def repeat_test(n, m, is_real_input, is_real_output, criteria, max_depth):
    fit_times = []
    predict_times = []

    for _ in range(num_average_time):
        fit_time, predict_time = get_time(n, m, is_real_input, is_real_output, criteria, max_depth)
        fit_times.append(fit_time)
        predict_times.append(predict_time)

    return fit_times, predict_times


def show_plot(is_real_input, is_real_output, criteria, depth, var_range, n=None, m=None):
    assert (n is None) != (m is None), "Either N or M should be fixed"

    fixed_val = m if n is None else n

    range_var = 'N' if n is None else 'M'
    fixed_var = 'N' if n is not None else 'M'

    avg_fit_times, avg_predict_times = [], []

    for var in var_range:
        if range_var == 'N':
            n = var
        else:
            m = var

        fit_times, predict_times = repeat_test(n, m, is_real_input, is_real_output, criteria, depth)

        avg_fit_times.append(np.mean(fit_times))
        avg_predict_times.append(np.mean(predict_times))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(var_range, avg_fit_times)
    ax[1].plot(var_range, avg_predict_times)

    ax[0].set_title('Fit')
    ax[0].set_xlabel(range_var)
    ax[0].set_ylabel('Time (s)')

    ax[1].set_title('Predict')
    ax[1].set_xlabel(range_var)
    ax[1].set_ylabel('Time (s)')

    input_type = 'real' if is_real_input else 'discrete'
    output_type = 'real' if is_real_output else 'discrete'

    fig.suptitle(f'Times for {input_type} input and {output_type} output '
                 f'where {fixed_var} = {fixed_val}, max_depth = {depth}')
    plt.show()


def main():
    for is_real_input in [True, False]:
        for is_real_output in [True, False]:
            criteria = 'mse' if is_real_output else 'entropy'

            show_plot(is_real_input, is_real_output, criteria, 10, range(5, 26, 5), m=10)
            show_plot(is_real_input, is_real_output, criteria, 1000, range(5, 26, 5), n=20)


if __name__ == '__main__':
    main()
