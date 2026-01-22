import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def reorder_missing_matrix(matrix, missing, reorder_rows=False, reorder_cols=False, is_diagonal=False, inplace=False, prefix=None):
    """
    Reorder the rows or columns of a time-varying matrix where all non-missing
    values are in the upper left corner of the matrix.

    Parameters
    ----------
    matrix : array_like
        The matrix to be reordered. Must have shape (n, m, nobs).
    missing : array_like of bool
        The vector of missing indices. Must have shape (k, nobs) where `k = n`
        if `reorder_rows is True` and `k = m` if `reorder_cols is True`.
    reorder_rows : bool, optional
        Whether or not the rows of the matrix should be re-ordered. Default
        is False.
    reorder_cols : bool, optional
        Whether or not the columns of the matrix should be re-ordered. Default
        is False.
    is_diagonal : bool, optional
        Whether or not the matrix is diagonal. If this is True, must also have
        `n = m`. Default is False.
    inplace : bool, optional
        Whether or not to reorder the matrix in-place.
    prefix : {'s', 'd', 'c', 'z'}, optional
        The Fortran prefix of the vector. Default is to automatically detect
        the dtype. This parameter should only be used with caution.

    Returns
    -------
    reordered_matrix : array_like
        The reordered matrix.
    """
    if prefix is None:
        prefix = find_best_blas_type((matrix,))[0]
    reorder = prefix_reorder_missing_matrix_map[prefix]
    if not inplace:
        matrix = np.copy(matrix, order='F')
    reorder(matrix, np.asfortranarray(missing), reorder_rows, reorder_cols, is_diagonal)
    return matrix