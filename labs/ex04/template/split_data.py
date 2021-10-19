# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio
    # ***************************************************
    x_shuffled = x.copy()
    y_shuffled = y.copy()
    p = np.random.permutation(len(x))
    x_shuffled, y_shuffled = x[p], y[p]

    ratio = int(ratio * x.shape[0])
    x_train, x_test = x_shuffled[:ratio], x_shuffled[ratio:]
    y_train, y_test = y_shuffled[:ratio], y_shuffled[ratio:]

    return x_train, x_test, y_train, y_test

