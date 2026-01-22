import itertools
import shutil
from functools import partial
from tempfile import mkdtemp
import numpy as np
import pytest
from scipy.cluster import hierarchy
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, ward_tree
from sklearn.cluster._agglomerative import (
from sklearn.cluster._hierarchical_fast import (
from sklearn.datasets import make_circles, make_moons
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import DistanceMetric
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import (
from sklearn.metrics.tests.test_dist_metrics import METRICS_DEFAULT_PARAMS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils._fast_dict import IntFloatDict
from sklearn.utils._testing import (
from sklearn.utils.fixes import LIL_CONTAINERS
def test_structured_linkage_tree():
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    mask[4:7, 4:7] = 0
    X = rng.randn(50, 100)
    connectivity = grid_to_graph(*mask.shape)
    for tree_builder in _TREE_BUILDERS.values():
        children, n_components, n_leaves, parent = tree_builder(X.T, connectivity=connectivity)
        n_nodes = 2 * X.shape[1] - 1
        assert len(children) + n_leaves == n_nodes
        with pytest.raises(ValueError):
            tree_builder(X.T, connectivity=np.ones((4, 4)))
        with pytest.raises(ValueError):
            tree_builder(X.T[:0], connectivity=connectivity)