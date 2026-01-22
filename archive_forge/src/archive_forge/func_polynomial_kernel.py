import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None], 'degree': [Interval(Real, 1, None, closed='left')], 'gamma': [Interval(Real, 0, None, closed='left'), None, Hidden(np.ndarray)], 'coef0': [Interval(Real, None, None, closed='neither')]}, prefer_skip_nested_validation=True)
def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y.

        K(X, Y) = (gamma <X, Y> + coef0) ^ degree

    Read more in the :ref:`User Guide <polynomial_kernel>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        A feature array.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    degree : float, default=3
        Kernel degree.

    gamma : float, default=None
        Coefficient of the vector inner product. If None, defaults to 1.0 / n_features.

    coef0 : float, default=1
        Constant offset added to scaled inner product.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The polynomial kernel.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import polynomial_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> polynomial_kernel(X, Y, degree=2)
    array([[1.     , 1.     ],
           [1.77..., 2.77...]])
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = safe_sparse_dot(X, Y.T, dense_output=True)
    K *= gamma
    K += coef0
    K **= degree
    return K