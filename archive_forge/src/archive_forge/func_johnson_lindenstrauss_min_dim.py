import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
@validate_params({'n_samples': ['array-like', Interval(Real, 1, None, closed='left')], 'eps': ['array-like', Interval(Real, 0, 1, closed='neither')]}, prefer_skip_nested_validation=True)
def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    """Find a 'safe' number of components to randomly project to.

    The distortion introduced by a random projection `p` only changes the
    distance between two points by a factor (1 +- eps) in a euclidean space
    with good probability. The projection `p` is an eps-embedding as defined
    by:

      (1 - eps) ||u - v||^2 < ||p(u) - p(v)||^2 < (1 + eps) ||u - v||^2

    Where u and v are any rows taken from a dataset of shape (n_samples,
    n_features), eps is in ]0, 1[ and p is a projection by a random Gaussian
    N(0, 1) matrix of shape (n_components, n_features) (or a sparse
    Achlioptas matrix).

    The minimum number of components to guarantee the eps-embedding is
    given by:

      n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)

    Note that the number of dimensions is independent of the original
    number of features but instead depends on the size of the dataset:
    the larger the dataset, the higher is the minimal dimensionality of
    an eps-embedding.

    Read more in the :ref:`User Guide <johnson_lindenstrauss>`.

    Parameters
    ----------
    n_samples : int or array-like of int
        Number of samples that should be an integer greater than 0. If an array
        is given, it will compute a safe number of components array-wise.

    eps : float or array-like of shape (n_components,), dtype=float,             default=0.1
        Maximum distortion rate in the range (0, 1) as defined by the
        Johnson-Lindenstrauss lemma. If an array is given, it will compute a
        safe number of components array-wise.

    Returns
    -------
    n_components : int or ndarray of int
        The minimal number of components to guarantee with good probability
        an eps-embedding with n_samples.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

    .. [2] `Sanjoy Dasgupta and Anupam Gupta, 1999,
           "An elementary proof of the Johnson-Lindenstrauss Lemma."
           <https://citeseerx.ist.psu.edu/doc_view/pid/95cd464d27c25c9c8690b378b894d337cdf021f9>`_

    Examples
    --------
    >>> from sklearn.random_projection import johnson_lindenstrauss_min_dim
    >>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
    663

    >>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
    array([    663,   11841, 1112658])

    >>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
    array([ 7894,  9868, 11841])
    """
    eps = np.asarray(eps)
    n_samples = np.asarray(n_samples)
    if np.any(eps <= 0.0) or np.any(eps >= 1):
        raise ValueError('The JL bound is defined for eps in ]0, 1[, got %r' % eps)
    if np.any(n_samples <= 0):
        raise ValueError('The JL bound is defined for n_samples greater than zero, got %r' % n_samples)
    denominator = eps ** 2 / 2 - eps ** 3 / 3
    return (4 * np.log(n_samples) / denominator).astype(np.int64)