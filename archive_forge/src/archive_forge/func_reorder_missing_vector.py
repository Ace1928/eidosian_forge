import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def reorder_missing_vector(vector, missing, inplace=False, prefix=None):
    """
    Reorder the elements of a time-varying vector where all non-missing
    values are in the first elements of the vector.

    Parameters
    ----------
    vector : array_like
        The vector to be reordered. Must have shape (n, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (n, nobs).
    inplace : bool, optional
        Whether or not to reorder the matrix in-place. Default is False.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    reordered_vector : array_like
        The reordered vector.
    """
    if prefix is None:
        prefix = find_best_blas_type((vector,))[0]
    reorder = prefix_reorder_missing_vector_map[prefix]
    if not inplace:
        vector = np.copy(vector, order='F')
    reorder(vector, np.asfortranarray(missing))
    return vector