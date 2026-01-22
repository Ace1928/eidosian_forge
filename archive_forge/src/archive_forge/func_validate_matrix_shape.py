import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def validate_matrix_shape(name, shape, nrows, ncols, nobs):
    """
    Validate the shape of a possibly time-varying matrix, or raise an exception

    Parameters
    ----------
    name : str
        The name of the matrix being validated (used in exception messages)
    shape : array_like
        The shape of the matrix to be validated. May be of size 2 or (if
        the matrix is time-varying) 3.
    nrows : int
        The expected number of rows.
    ncols : int
        The expected number of columns.
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying matrix)

    Raises
    ------
    ValueError
        If the matrix is not of the desired shape.
    """
    ndim = len(shape)
    if ndim not in [2, 3]:
        raise ValueError('Invalid value for %s matrix. Requires a 2- or 3-dimensional array, got %d dimensions' % (name, ndim))
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s matrix: requires %d rows, got %d' % (name, nrows, shape[0]))
    if not shape[1] == ncols:
        raise ValueError('Invalid dimensions for %s matrix: requires %d columns, got %d' % (name, ncols, shape[1]))
    if nobs is None and (not (ndim == 2 or shape[-1] == 1)):
        raise ValueError('Invalid dimensions for %s matrix: time-varying matrices cannot be given unless `nobs` is specified (implicitly when a dataset is bound or else set explicity)' % name)
    if ndim == 3 and nobs is not None and (not shape[-1] in [1, nobs]):
        raise ValueError('Invalid dimensions for time-varying %s matrix. Requires shape (*,*,%d), got %s' % (name, nobs, str(shape)))