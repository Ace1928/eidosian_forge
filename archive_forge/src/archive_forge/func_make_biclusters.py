import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
@validate_params({'shape': [tuple], 'n_clusters': [Interval(Integral, 1, None, closed='left')], 'noise': [Interval(Real, 0, None, closed='left')], 'minval': [Interval(Real, None, None, closed='neither')], 'maxval': [Interval(Real, None, None, closed='neither')], 'shuffle': ['boolean'], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_biclusters(shape, n_clusters, *, noise=0.0, minval=10, maxval=100, shuffle=True, random_state=None):
    """Generate a constant block diagonal structure array for biclustering.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    shape : tuple of shape (n_rows, n_cols)
        The shape of the result.

    n_clusters : int
        The number of biclusters.

    noise : float, default=0.0
        The standard deviation of the gaussian noise.

    minval : float, default=10
        Minimum value of a bicluster.

    maxval : float, default=100
        Maximum value of a bicluster.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape `shape`
        The generated array.

    rows : ndarray of shape (n_clusters, X.shape[0])
        The indicators for cluster membership of each row.

    cols : ndarray of shape (n_clusters, X.shape[1])
        The indicators for cluster membership of each column.

    See Also
    --------
    make_checkerboard: Generate an array with block checkerboard structure for
        biclustering.

    References
    ----------

    .. [1] Dhillon, I. S. (2001, August). Co-clustering documents and
        words using bipartite spectral graph partitioning. In Proceedings
        of the seventh ACM SIGKDD international conference on Knowledge
        discovery and data mining (pp. 269-274). ACM.
    """
    generator = check_random_state(random_state)
    n_rows, n_cols = shape
    consts = generator.uniform(minval, maxval, n_clusters)
    row_sizes = generator.multinomial(n_rows, np.repeat(1.0 / n_clusters, n_clusters))
    col_sizes = generator.multinomial(n_cols, np.repeat(1.0 / n_clusters, n_clusters))
    row_labels = np.hstack([np.repeat(val, rep) for val, rep in zip(range(n_clusters), row_sizes)])
    col_labels = np.hstack([np.repeat(val, rep) for val, rep in zip(range(n_clusters), col_sizes)])
    result = np.zeros(shape, dtype=np.float64)
    for i in range(n_clusters):
        selector = np.outer(row_labels == i, col_labels == i)
        result[selector] += consts[i]
    if noise > 0:
        result += generator.normal(scale=noise, size=result.shape)
    if shuffle:
        result, row_idx, col_idx = _shuffle(result, random_state)
        row_labels = row_labels[row_idx]
        col_labels = col_labels[col_idx]
    rows = np.vstack([row_labels == c for c in range(n_clusters)])
    cols = np.vstack([col_labels == c for c in range(n_clusters)])
    return (result, rows, cols)