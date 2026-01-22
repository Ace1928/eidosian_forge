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
def test_dist_threshold_invalid_parameters():
    X = [[0], [1]]
    with pytest.raises(ValueError, match='Exactly one of '):
        AgglomerativeClustering(n_clusters=None, distance_threshold=None).fit(X)
    with pytest.raises(ValueError, match='Exactly one of '):
        AgglomerativeClustering(n_clusters=2, distance_threshold=1).fit(X)
    X = [[0], [1]]
    with pytest.raises(ValueError, match='compute_full_tree must be True if'):
        AgglomerativeClustering(n_clusters=None, distance_threshold=1, compute_full_tree=False).fit(X)