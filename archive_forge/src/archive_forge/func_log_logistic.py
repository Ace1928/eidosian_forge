import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
@deprecated('The function `log_logistic` is deprecated and will be removed in 1.6. Use `-np.logaddexp(0, -x)` instead.')
def log_logistic(X, out=None):
    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.

    This implementation is numerically stable and uses `-np.logaddexp(0, -x)`.

    For the ordinary logistic function, use ``scipy.special.expit``.

    Parameters
    ----------
    X : array-like of shape (M, N) or (M,)
        Argument to the logistic function.

    out : array-like of shape (M, N) or (M,), default=None
        Preallocated output array.

    Returns
    -------
    out : ndarray of shape (M, N) or (M,)
        Log of the logistic function evaluated at every point in x.

    Notes
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    X = check_array(X, dtype=np.float64, ensure_2d=False)
    if out is None:
        out = np.empty_like(X)
    np.logaddexp(0, -X, out=out)
    out *= -1
    return out