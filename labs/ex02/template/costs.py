# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np


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
