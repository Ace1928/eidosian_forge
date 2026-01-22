from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import csgraph, issparse
from ...base import BaseEstimator, ClusterMixin, _fit_context
from ...metrics import pairwise_distances
from ...metrics._dist_metrics import DistanceMetric
from ...neighbors import BallTree, KDTree, NearestNeighbors
from ...utils._param_validation import Interval, StrOptions
from ...utils.validation import _allclose_dense_sparse, _assert_all_finite
from ._linkage import (
from ._reachability import mutual_reachability_graph
from ._tree import HIERARCHY_dtype, labelling_at_cut, tree_to_labels
def remap_single_linkage_tree(tree, internal_to_raw, non_finite):
    """
    Takes an internal single_linkage_tree structure and adds back in a set of points
    that were initially detected as non-finite and returns that new tree.
    These points will all be merged into the final node at np.inf distance and
    considered noise points.

    Parameters
    ----------
    tree : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype
        The single-linkage tree tree (dendrogram) built from the MST.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index
    non_finite : ndarray
        Boolean array of which entries in the raw data are non-finite
    """
    finite_count = len(internal_to_raw)
    outlier_count = len(non_finite)
    for i, _ in enumerate(tree):
        left = tree[i]['left_node']
        right = tree[i]['right_node']
        if left < finite_count:
            tree[i]['left_node'] = internal_to_raw[left]
        else:
            tree[i]['left_node'] = left + outlier_count
        if right < finite_count:
            tree[i]['right_node'] = internal_to_raw[right]
        else:
            tree[i]['right_node'] = right + outlier_count
    outlier_tree = np.zeros(len(non_finite), dtype=HIERARCHY_dtype)
    last_cluster_id = max(tree[tree.shape[0] - 1]['left_node'], tree[tree.shape[0] - 1]['right_node'])
    last_cluster_size = tree[tree.shape[0] - 1]['cluster_size']
    for i, outlier in enumerate(non_finite):
        outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)
        last_cluster_id += 1
        last_cluster_size += 1
    tree = np.concatenate([tree, outlier_tree])
    return tree