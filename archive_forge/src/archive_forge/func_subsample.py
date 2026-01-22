from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def subsample(*data, n=10000, seed=None):
    """Subsample the number of points in a dataset.

    Selects a random subset of (optionally multiple) datasets.
    Helpful for plotting, or for methods with computational
    constraints.

    Parameters
    ----------
    data : array-like, shape=[n_samples, any]
        Input data. Any number of datasets can be passed at once,
        so long as `n_samples` remains the same.
    n : int, optional (default: 10000)
        Number of samples to retain. Must be less than `n_samples`.
    seed : int, optional (default: None)
        Random seed

    Examples
    --------
    data_subsample, labels_subsample = scprep.utils.subsample(data, labels, n=1000)
    """
    N = data[0].shape[0]
    if len(data) > 1:
        _check_rows_compatible(*data)
    if N < n:
        raise ValueError('Expected n ({}) <= n_samples ({})'.format(n, N))
    np.random.seed(seed)
    select_idx = np.isin(np.arange(N), np.random.choice(N, n, replace=False))
    data = [select_rows(d, idx=select_idx) for d in data]
    return tuple(data) if len(data) > 1 else data[0]