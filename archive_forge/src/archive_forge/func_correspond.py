import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def correspond(Z, Y):
    """
    Check for correspondence between linkage and condensed distance matrices.

    They must have the same number of original observations for
    the check to succeed.

    This function is useful as a sanity check in algorithms that make
    extensive use of linkage and distance matrices that must
    correspond to the same set of original observations.

    Parameters
    ----------
    Z : array_like
        The linkage matrix to check for correspondence.
    Y : array_like
        The condensed distance matrix to check for correspondence.

    Returns
    -------
    b : bool
        A boolean indicating whether the linkage matrix and distance
        matrix could possibly correspond to one another.

    See Also
    --------
    linkage : for a description of what a linkage matrix is.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, correspond
    >>> from scipy.spatial.distance import pdist

    This method can be used to check if a given linkage matrix ``Z`` has been
    obtained from the application of a cluster method over a dataset ``X``:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]
    >>> X_condensed = pdist(X)
    >>> Z = ward(X_condensed)

    Here, we can compare ``Z`` and ``X`` (in condensed form):

    >>> correspond(Z, X_condensed)
    True

    """
    is_valid_linkage(Z, throw=True)
    distance.is_valid_y(Y, throw=True)
    xp = array_namespace(Z, Y)
    Z = as_xparray(Z, order='c', xp=xp)
    Y = as_xparray(Y, order='c', xp=xp)
    return distance.num_obs_y(Y) == num_obs_linkage(Z)