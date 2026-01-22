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
def test_linkage_misc():
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, 5))
    with pytest.raises(ValueError):
        linkage_tree(X, linkage='foo')
    with pytest.raises(ValueError):
        linkage_tree(X, connectivity=np.ones((4, 4)))
    FeatureAgglomeration().fit(X)
    dis = cosine_distances(X)
    res = linkage_tree(dis, affinity='precomputed')
    assert_array_equal(res[0], linkage_tree(X, affinity='cosine')[0])
    res = linkage_tree(X, affinity=manhattan_distances)
    assert_array_equal(res[0], linkage_tree(X, affinity='manhattan')[0])