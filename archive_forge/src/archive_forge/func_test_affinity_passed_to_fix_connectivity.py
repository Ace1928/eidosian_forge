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
def test_affinity_passed_to_fix_connectivity():
    size = 2
    rng = np.random.RandomState(0)
    X = rng.randn(size, size)
    mask = np.array([True, False, False, True])
    connectivity = grid_to_graph(n_x=size, n_y=size, mask=mask, return_as=np.ndarray)

    class FakeAffinity:

        def __init__(self):
            self.counter = 0

        def increment(self, *args, **kwargs):
            self.counter += 1
            return self.counter
    fa = FakeAffinity()
    linkage_tree(X, connectivity=connectivity, affinity=fa.increment)
    assert fa.counter == 3