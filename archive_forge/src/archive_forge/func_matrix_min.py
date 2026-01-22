from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def matrix_min(data):
    """Get the minimum value from a data matrix.

    Pandas SparseDataFrame does not handle np.min.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    minimum : float
        Minimum entry in `data`.
    """
    if is_SparseDataFrame(data):
        data = [np.min(data[col]) for col in data.columns]
    elif is_sparse_dataframe(data):
        data = [sparse_series_min(data[col]) for col in data.columns]
    elif isinstance(data, pd.DataFrame):
        data = np.min(data)
    elif isinstance(data, sparse.lil_matrix):
        data = [np.min(d) for d in data.data] + [0]
    elif isinstance(data, sparse.dok_matrix):
        data = list(data.values()) + [0]
    elif isinstance(data, sparse.dia_matrix):
        data = [np.min(data.data), 0]
    return np.min(data)