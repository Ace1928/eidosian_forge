import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def leaders(Z, T):
    """
    Return the root nodes in a hierarchical clustering.

    Returns the root nodes in a hierarchical clustering corresponding
    to a cut defined by a flat cluster assignment vector ``T``. See
    the ``fcluster`` function for more information on the format of ``T``.

    For each flat cluster :math:`j` of the :math:`k` flat clusters
    represented in the n-sized flat cluster assignment vector ``T``,
    this function finds the lowest cluster node :math:`i` in the linkage
    tree Z, such that:

      * leaf descendants belong only to flat cluster j
        (i.e., ``T[p]==j`` for all :math:`p` in :math:`S(i)`, where
        :math:`S(i)` is the set of leaf ids of descendant leaf nodes
        with cluster node :math:`i`)

      * there does not exist a leaf that is not a descendant with
        :math:`i` that also belongs to cluster :math:`j`
        (i.e., ``T[q]!=j`` for all :math:`q` not in :math:`S(i)`). If
        this condition is violated, ``T`` is not a valid cluster
        assignment vector, and an exception will be thrown.

    Parameters
    ----------
    Z : ndarray
        The hierarchical clustering encoded as a matrix. See
        `linkage` for more information.
    T : ndarray
        The flat cluster assignment vector.

    Returns
    -------
    L : ndarray
        The leader linkage node id's stored as a k-element 1-D array,
        where ``k`` is the number of flat clusters found in ``T``.

        ``L[j]=i`` is the linkage cluster node id that is the
        leader of flat cluster with id M[j]. If ``i < n``, ``i``
        corresponds to an original observation, otherwise it
        corresponds to a non-singleton cluster.
    M : ndarray
        The leader linkage node id's stored as a k-element 1-D array, where
        ``k`` is the number of flat clusters found in ``T``. This allows the
        set of flat cluster ids to be any arbitrary set of ``k`` integers.

        For example: if ``L[3]=2`` and ``M[3]=8``, the flat cluster with
        id 8's leader is linkage node 2.

    See Also
    --------
    fcluster : for the creation of flat cluster assignments.

    Examples
    --------
    >>> from scipy.cluster.hierarchy import ward, fcluster, leaders
    >>> from scipy.spatial.distance import pdist

    Given a linkage matrix ``Z`` - obtained after apply a clustering method
    to a dataset ``X`` - and a flat cluster assignment array ``T``:

    >>> X = [[0, 0], [0, 1], [1, 0],
    ...      [0, 4], [0, 3], [1, 4],
    ...      [4, 0], [3, 0], [4, 1],
    ...      [4, 4], [3, 4], [4, 3]]

    >>> Z = ward(pdist(X))
    >>> Z
    array([[ 0.        ,  1.        ,  1.        ,  2.        ],
           [ 3.        ,  4.        ,  1.        ,  2.        ],
           [ 6.        ,  7.        ,  1.        ,  2.        ],
           [ 9.        , 10.        ,  1.        ,  2.        ],
           [ 2.        , 12.        ,  1.29099445,  3.        ],
           [ 5.        , 13.        ,  1.29099445,  3.        ],
           [ 8.        , 14.        ,  1.29099445,  3.        ],
           [11.        , 15.        ,  1.29099445,  3.        ],
           [16.        , 17.        ,  5.77350269,  6.        ],
           [18.        , 19.        ,  5.77350269,  6.        ],
           [20.        , 21.        ,  8.16496581, 12.        ]])

    >>> T = fcluster(Z, 3, criterion='distance')
    >>> T
    array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], dtype=int32)

    `scipy.cluster.hierarchy.leaders` returns the indices of the nodes
    in the dendrogram that are the leaders of each flat cluster:

    >>> L, M = leaders(Z, T)
    >>> L
    array([16, 17, 18, 19], dtype=int32)

    (remember that indices 0-11 point to the 12 data points in ``X``,
    whereas indices 12-22 point to the 11 rows of ``Z``)

    `scipy.cluster.hierarchy.leaders` also returns the indices of
    the flat clusters in ``T``:

    >>> M
    array([1, 2, 3, 4], dtype=int32)

    """
    xp = array_namespace(Z, T)
    Z = as_xparray(Z, order='C', dtype=xp.float64)
    T = as_xparray(T, order='C')
    is_valid_linkage(Z, throw=True, name='Z')
    if T.dtype != xp.int32:
        raise TypeError('T must be a 1-D array of dtype int32.')
    if T.shape[0] != Z.shape[0] + 1:
        raise ValueError('Mismatch: len(T)!=Z.shape[0] + 1.')
    n_clusters = int(xp.unique_values(T).shape[0])
    n_obs = int(Z.shape[0] + 1)
    L = np.zeros(n_clusters, dtype=np.int32)
    M = np.zeros(n_clusters, dtype=np.int32)
    Z = np.asarray(Z)
    T = np.asarray(T, dtype=np.int32)
    s = _hierarchy.leaders(Z, T, L, M, n_clusters, n_obs)
    if s >= 0:
        raise ValueError('T is not a valid assignment vector. Error found when examining linkage node %d (< 2n-1).' % s)
    L, M = (xp.asarray(L), xp.asarray(M))
    return (L, M)