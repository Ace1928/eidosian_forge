import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from ..utils.fixes import _sparse_min_max, _sparse_nan_min_max
from ..utils.validation import _check_sample_weight
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
from .sparsefuncs_fast import (
def csc_median_axis_0(X):
    """Find the median across axis 0 of a CSC matrix.

    It is equivalent to doing np.median(X, axis=0).

    Parameters
    ----------
    X : sparse matrix of shape (n_samples, n_features)
        Input data. It should be of CSC format.

    Returns
    -------
    median : ndarray of shape (n_features,)
        Median.
    """
    if not (sp.issparse(X) and X.format == 'csc'):
        raise TypeError('Expected matrix of CSC format, got %s' % X.format)
    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.zeros(n_features)
    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        data = np.copy(X.data[start:end])
        nz = n_samples - data.size
        median[f_ind] = _get_median(data, nz)
    return median