# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from costs import compute_loss
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w, sub=False):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    e = y - tx @ np.asarray(w)
    N = tx.shape[0]
    if sub:
        gradient = np.mean(np.array([-np.sign(e), -(np.sign(e)) * tx[:, 1]]), axis=1)
    else:
        gradient = - 1 / N * tx.T @ e
    return gradient
    # ***************************************************


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, mae=False):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        small_y, small_tx = batch_iter(y, tx, batch_size).__next__()
        small_y, small_tx = np.asarray(small_y), np.asarray(small_tx)
        loss = compute_loss(small_y, small_tx, w)
        gradient = compute_stoch_gradient(small_y, small_tx, w, mae)

        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
