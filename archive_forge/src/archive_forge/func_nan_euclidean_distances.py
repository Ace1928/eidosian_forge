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
@validate_params({'X': ['array-like'], 'Y': ['array-like', None], 'squared': ['boolean'], 'missing_values': [MissingValues(numeric_only=True)], 'copy': ['boolean']}, prefer_skip_nested_validation=True)
def nan_euclidean_distances(X, Y=None, *, squared=False, missing_values=np.nan, copy=True):
    """Calculate the euclidean distances in the presence of missing values.

    Compute the euclidean distance between each pair of samples in X and Y,
    where Y=X is assumed if Y=None. When calculating the distance between a
    pair of samples, this formulation ignores feature coordinates with a
    missing value in either sample and scales up the weight of the remaining
    coordinates:

        dist(x,y) = sqrt(weight * sq. distance from present coordinates)
        where,
        weight = Total # of coordinates / # of present coordinates

    For example, the distance between ``[3, na, na, 6]`` and ``[1, na, 4, 5]``
    is:

        .. math::
            \\sqrt{\\frac{4}{2}((3-1)^2 + (6-5)^2)}

    If all the coordinates are missing or if there are no common present
    coordinates then NaN is returned for that pair.

    Read more in the :ref:`User Guide <metrics>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        An array where each row is a sample and each column is a feature.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        An array where each row is a sample and each column is a feature.
        If `None`, method uses `Y=X`.

    squared : bool, default=False
        Return squared Euclidean distances.

    missing_values : np.nan, float or int, default=np.nan
        Representation of missing value.

    copy : bool, default=True
        Make and use a deep copy of X and Y (if Y exists).

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the distances between the row vectors of `X`
        and the row vectors of `Y`.

    See Also
    --------
    paired_distances : Distances between pairs of elements of X and Y.

    References
    ----------
    * John K. Dixon, "Pattern Recognition with Partly Missing Data",
      IEEE Transactions on Systems, Man, and Cybernetics, Volume: 9, Issue:
      10, pp. 617 - 621, Oct. 1979.
      http://ieeexplore.ieee.org/abstract/document/4310090/

    Examples
    --------
    >>> from sklearn.metrics.pairwise import nan_euclidean_distances
    >>> nan = float("NaN")
    >>> X = [[0, 1], [1, nan]]
    >>> nan_euclidean_distances(X, X) # distance between rows of X
    array([[0.        , 1.41421356],
           [1.41421356, 0.        ]])

    >>> # get distance to origin
    >>> nan_euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    """
    force_all_finite = 'allow-nan' if is_scalar_nan(missing_values) else True
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False, force_all_finite=force_all_finite, copy=copy)
    missing_X = _get_mask(X, missing_values)
    missing_Y = missing_X if Y is X else _get_mask(Y, missing_values)
    X[missing_X] = 0
    Y[missing_Y] = 0
    distances = euclidean_distances(X, Y, squared=True)
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)
    np.clip(distances, 0, None, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0.0)
    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)
    distances[present_count == 0] = np.nan
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X.shape[1]
    if not squared:
        np.sqrt(distances, out=distances)
    return distances