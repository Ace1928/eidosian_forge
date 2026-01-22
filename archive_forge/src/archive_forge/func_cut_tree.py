import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def cut_tree(Z, n_clusters=None, height=None):
    """
    Given a linkage matrix Z, return the cut tree.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.
    n_clusters : array_like, optional
        Number of clusters in the tree at the cut point.
    height : array_like, optional
        The height at which to cut the tree. Only possible for ultrametric
        trees.

    Returns
    -------
    cutree : array
        An array indicating group membership at each agglomeration step. I.e.,
        for a full cut tree, in the first column each data point is in its own
        cluster. At the next step, two nodes are merged. Finally, all
        singleton and non-singleton clusters are in one group. If `n_clusters`
        or `height` are given, the columns correspond to the columns of
        `n_clusters` or `height`.

    Examples
    --------
    >>> from scipy import cluster
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> X = rng.random((50, 4))
    >>> Z = cluster.hierarchy.ward(X)
    >>> cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
    >>> cutree[:10]
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3],
           [3, 4],
           [2, 2],
           [0, 0],
           [1, 5],
           [3, 6],
           [4, 7]])  # random

    """
    xp = array_namespace(Z)
    nobs = num_obs_linkage(Z)
    nodes = _order_cluster_tree(Z)
    if height is not None and n_clusters is not None:
        raise ValueError('At least one of either height or n_clusters must be None')
    elif height is None and n_clusters is None:
        cols_idx = xp.arange(nobs)
    elif height is not None:
        height = xp.asarray(height)
        heights = xp.asarray([x.dist for x in nodes])
        cols_idx = xp.searchsorted(heights, height)
    else:
        n_clusters = xp.asarray(n_clusters)
        cols_idx = nobs - xp.searchsorted(xp.arange(nobs), n_clusters)
    try:
        n_cols = len(cols_idx)
    except TypeError:
        n_cols = 1
        cols_idx = xp.asarray([cols_idx])
    groups = xp.zeros((n_cols, nobs), dtype=xp.int64)
    last_group = xp.arange(nobs)
    if 0 in cols_idx:
        groups[0] = last_group
    for i, node in enumerate(nodes):
        idx = node.pre_order()
        this_group = copy(last_group, xp=xp)
        this_group[idx] = xp.min(last_group[idx])
        this_group[this_group > xp.max(last_group[idx])] -= 1
        if i + 1 in cols_idx:
            groups[np.nonzero(i + 1 == cols_idx)[0]] = this_group
        last_group = this_group
    return groups.T