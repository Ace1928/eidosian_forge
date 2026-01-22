import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def optimal_leaf_ordering(Z, y, metric='euclidean'):
    """
    Given a linkage matrix Z and distance, reorder the cut tree.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a linkage matrix. See
        `linkage` for more information on the return structure and
        algorithm.
    y : ndarray
        The condensed distance matrix from which Z was generated.
        Alternatively, a collection of m observation vectors in n
        dimensions may be passed as an m by n array.
    metric : str or function, optional
        The distance metric to use in the case that y is a collection of
        observation vectors; ignored otherwise. See the ``pdist``
        function for a list of valid distance metrics. A custom distance
        function can also be used.

    Returns
    -------
    Z_ordered : ndarray
        A copy of the linkage matrix Z, reordered to minimize the distance
        between adjacent leaves.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> X = rng.standard_normal((10, 10))
    >>> Z = hierarchy.ward(X)
    >>> hierarchy.leaves_list(Z)
    array([0, 3, 1, 9, 2, 5, 7, 4, 6, 8], dtype=int32)
    >>> hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
    array([3, 0, 2, 5, 7, 4, 8, 6, 9, 1], dtype=int32)

    """
    xp = array_namespace(Z, y)
    Z = as_xparray(Z, order='C', xp=xp)
    is_valid_linkage(Z, throw=True, name='Z')
    y = as_xparray(y, order='C', dtype=xp.float64, xp=xp)
    if y.ndim == 1:
        distance.is_valid_y(y, throw=True, name='y')
    elif y.ndim == 2:
        if y.shape[0] == y.shape[1] and np.allclose(np.diag(y), 0) and np.all(y >= 0) and np.allclose(y, y.T):
            warnings.warn('The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix', ClusterWarning, stacklevel=2)
        y = distance.pdist(y, metric)
        y = xp.asarray(y)
    else:
        raise ValueError('`y` must be 1 or 2 dimensional.')
    if not xp.all(xp.isfinite(y)):
        raise ValueError('The condensed distance matrix must contain only finite values.')
    Z = np.asarray(Z)
    y = np.asarray(y)
    return xp.asarray(_optimal_leaf_ordering.optimal_leaf_ordering(Z, y))