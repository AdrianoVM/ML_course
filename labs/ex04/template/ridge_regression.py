# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    lambda_i = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w_prime = np.linalg.solve(tx.T @ tx + lambda_i, tx.T @ y)
    mse = compute_mse(y, tx, w_prime)
    return w_prime, mse
