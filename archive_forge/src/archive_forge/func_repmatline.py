from __future__ import division
import sys
import importlib
import logging
import functools
import pkgutil
import io
import numpy as np
from scipy import sparse
import scipy.io
def repmatline(A, ncol=1, nrow=1):
    """
    Repeat the matrix A in a specific manner.

    Parameters
    ----------
    A : ndarray
    ncol : int
        default is 1
    nrow : int
        default is 1

    Returns
    -------
    Ar : ndarray

    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.array([[1, 2], [3, 4]])
    >>> x
    array([[1, 2],
           [3, 4]])
    >>> utils.repmatline(x, nrow=2, ncol=3)
    array([[1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4],
           [3, 3, 3, 4, 4, 4]])

    """
    if ncol < 1 or nrow < 1:
        raise ValueError('The number of lines and rows must be greater or equal to one, or you will get an empty array.')
    return np.repeat(np.repeat(A, ncol, axis=1), nrow, axis=0)