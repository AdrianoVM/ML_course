# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns optimal weights, MSE
    # ***************************************************
    w_prime = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_mse(y, tx, w_prime)
    return w_prime, mse
