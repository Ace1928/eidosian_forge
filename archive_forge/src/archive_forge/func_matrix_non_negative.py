from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_non_negative(data, allow_equal=True):
    """Check if all values in a matrix are non-negative.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    allow_equal : bool, optional (default: True)
        If True, min(data) can be equal to 0

    Returns
    -------
    is_non_negative : bool
    """
    return matrix_min(data) >= 0 if allow_equal else matrix_min(data) > 0