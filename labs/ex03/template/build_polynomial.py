# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.power(np.repeat(np.expand_dims(x, axis=1), degree+1, axis=1), np.arange(degree+1))
    return poly
