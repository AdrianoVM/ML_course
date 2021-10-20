# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""

    w_prime = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = compute_loss(y, tx, w_prime)
    return w_prime, mse


def compute_loss(y, tx, w, mae=False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ***************************************************
    N = tx.shape[0]
    if mae:
        print(w)
        e = y - tx @ np.asarray(w)
        loss = np.sum(abs(e)) / (2 * N)
    else:
        e = y - tx @ np.asarray(w)
        loss = (e @ e.T) / (2 * N)
    return loss
