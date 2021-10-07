# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
from costs import compute_loss


def compute_gradient(y, tx, w, sub=False):
    """Compute the gradient."""
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


def gradient_descent(y, tx, initial_w, max_iters, gamma, mae=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        loss = compute_loss(y, tx, w, mae)
        gradient = compute_gradient(y, tx, w, mae)
        # ***************************************************
        # ***************************************************
        # INSERT YOUR CODE HERE
        w = w - gamma * gradient
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
