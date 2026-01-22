import warnings
from heapq import heapify, heappop, heappush, heappushpop
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from ..base import (
from ..metrics import DistanceMetric
from ..metrics._dist_metrics import METRIC_MAPPING64
from ..metrics.pairwise import _VALID_METRICS, paired_distances
from ..utils import check_array
from ..utils._fast_dict import IntFloatDict
from ..utils._param_validation import (
from ..utils.graph import _fix_connected_components
from ..utils.validation import check_memory
from . import _hierarchical_fast as _hierarchical  # type: ignore
from ._feature_agglomeration import AgglomerationTransform
def linkage_tree(X, connectivity=None, n_clusters=None, linkage='complete', affinity='euclidean', return_distance=False):
    """Linkage agglomerative clustering based on a Feature matrix.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix representing `n_samples` samples to be clustered.

    connectivity : sparse matrix, default=None
        Connectivity matrix. Defines for each sample the neighboring samples
        following a given structure of the data. The matrix is assumed to
        be symmetric and only the upper triangular half is used.
        Default is `None`, i.e, the Ward algorithm is unstructured.

    n_clusters : int, default=None
        Stop early the construction of the tree at `n_clusters`. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    linkage : {"average", "complete", "single"}, default="complete"
        Which linkage criteria to use. The linkage criterion determines which
        distance to use between sets of observation.
            - "average" uses the average of the distances of each observation of
              the two sets.
            - "complete" or maximum linkage uses the maximum distances between
              all observations of the two sets.
            - "single" uses the minimum of the distances between all
              observations of the two sets.

    affinity : str or callable, default='euclidean'
        Which metric to use. Can be 'euclidean', 'manhattan', or any
        distance known to paired distance (see metric.pairwise).

    return_distance : bool, default=False
        Whether or not to return the distances between the clusters.

    Returns
    -------
    children : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    n_connected_components : int
        The number of connected components in the graph.

    n_leaves : int
        The number of leaves in the tree.

    parents : ndarray of shape (n_nodes, ) or None
        The parent of each node. Only returned when a connectivity matrix
        is specified, elsewhere 'None' is returned.

    distances : ndarray of shape (n_nodes-1,)
        Returned when `return_distance` is set to `True`.

        distances[i] refers to the distance between children[i][0] and
        children[i][1] when they are merged.

    See Also
    --------
    ward_tree : Hierarchical clustering with ward linkage.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    n_samples, n_features = X.shape
    linkage_choices = {'complete': _hierarchical.max_merge, 'average': _hierarchical.average_merge, 'single': None}
    try:
        join_func = linkage_choices[linkage]
    except KeyError as e:
        raise ValueError('Unknown linkage option, linkage should be one of %s, but %s was given' % (linkage_choices.keys(), linkage)) from e
    if affinity == 'cosine' and np.any(~np.any(X, axis=1)):
        raise ValueError('Cosine affinity cannot be used when X contains zero vectors')
    if connectivity is None:
        from scipy.cluster import hierarchy
        if n_clusters is not None:
            warnings.warn('Partial build of the tree is implemented only for structured clustering (i.e. with explicit connectivity). The algorithm will build the full tree and only retain the lower branches required for the specified number of clusters', stacklevel=2)
        if affinity == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError(f'Distance matrix should be square, got matrix of shape {X.shape}')
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        elif affinity == 'l2':
            affinity = 'euclidean'
        elif affinity in ('l1', 'manhattan'):
            affinity = 'cityblock'
        elif callable(affinity):
            X = affinity(X)
            i, j = np.triu_indices(X.shape[0], k=1)
            X = X[i, j]
        if linkage == 'single' and affinity != 'precomputed' and (not callable(affinity)) and (affinity in METRIC_MAPPING64):
            dist_metric = DistanceMetric.get_metric(affinity)
            X = np.ascontiguousarray(X, dtype=np.double)
            mst = _hierarchical.mst_linkage_core(X, dist_metric)
            mst = mst[np.argsort(mst.T[2], kind='mergesort'), :]
            out = _hierarchical.single_linkage_label(mst)
        else:
            out = hierarchy.linkage(X, method=linkage, metric=affinity)
        children_ = out[:, :2].astype(int, copy=False)
        if return_distance:
            distances = out[:, 2]
            return (children_, 1, n_samples, None, distances)
        return (children_, 1, n_samples, None)
    connectivity, n_connected_components = _fix_connectivity(X, connectivity, affinity=affinity)
    connectivity = connectivity.tocoo()
    diag_mask = connectivity.row != connectivity.col
    connectivity.row = connectivity.row[diag_mask]
    connectivity.col = connectivity.col[diag_mask]
    connectivity.data = connectivity.data[diag_mask]
    del diag_mask
    if affinity == 'precomputed':
        distances = X[connectivity.row, connectivity.col].astype(np.float64, copy=False)
    else:
        distances = paired_distances(X[connectivity.row], X[connectivity.col], metric=affinity)
    connectivity.data = distances
    if n_clusters is None:
        n_nodes = 2 * n_samples - 1
    else:
        assert n_clusters <= n_samples
        n_nodes = 2 * n_samples - n_clusters
    if linkage == 'single':
        return _single_linkage_tree(connectivity, n_samples, n_nodes, n_clusters, n_connected_components, return_distance)
    if return_distance:
        distances = np.empty(n_nodes - n_samples)
    A = np.empty(n_nodes, dtype=object)
    inertia = list()
    connectivity = connectivity.tolil()
    for ind, (data, row) in enumerate(zip(connectivity.data, connectivity.rows)):
        A[ind] = IntFloatDict(np.asarray(row, dtype=np.intp), np.asarray(data, dtype=np.float64))
        inertia.extend((_hierarchical.WeightedEdge(d, ind, r) for r, d in zip(row, data) if r < ind))
    del connectivity
    heapify(inertia)
    parent = np.arange(n_nodes, dtype=np.intp)
    used_node = np.ones(n_nodes, dtype=np.intp)
    children = []
    for k in range(n_samples, n_nodes):
        while True:
            edge = heappop(inertia)
            if used_node[edge.a] and used_node[edge.b]:
                break
        i = edge.a
        j = edge.b
        if return_distance:
            distances[k - n_samples] = edge.weight
        parent[i] = parent[j] = k
        children.append((i, j))
        n_i = used_node[i]
        n_j = used_node[j]
        used_node[k] = n_i + n_j
        used_node[i] = used_node[j] = False
        coord_col = join_func(A[i], A[j], used_node, n_i, n_j)
        for col, d in coord_col:
            A[col].append(k, d)
            heappush(inertia, _hierarchical.WeightedEdge(d, k, col))
        A[k] = coord_col
        A[i] = A[j] = 0
    n_leaves = n_samples
    children = np.array(children)[:, ::-1]
    if return_distance:
        return (children, n_connected_components, n_leaves, parent, distances)
    return (children, n_connected_components, n_leaves, parent)