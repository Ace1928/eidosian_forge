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
@pytest.mark.parametrize('n_clusters, distance_threshold', [(None, 0.5), (10, None)])
@pytest.mark.parametrize('compute_distances', [True, False])
@pytest.mark.parametrize('linkage', ['ward', 'complete', 'average', 'single'])
def test_agglomerative_clustering_distances(n_clusters, compute_distances, distance_threshold, linkage):
    rng = np.random.RandomState(0)
    mask = np.ones([10, 10], dtype=bool)
    n_samples = 100
    X = rng.randn(n_samples, 50)
    connectivity = grid_to_graph(*mask.shape)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage=linkage, distance_threshold=distance_threshold, compute_distances=compute_distances)
    clustering.fit(X)
    if compute_distances or distance_threshold is not None:
        assert hasattr(clustering, 'distances_')
        n_children = clustering.children_.shape[0]
        n_nodes = n_children + 1
        assert clustering.distances_.shape == (n_nodes - 1,)
    else:
        assert not hasattr(clustering, 'distances_')