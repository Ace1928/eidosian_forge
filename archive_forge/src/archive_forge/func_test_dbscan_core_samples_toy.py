import pickle
import warnings
import numpy as np
import pytest
from scipy.spatial import distance
from sklearn.cluster import DBSCAN, dbscan
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
@pytest.mark.parametrize('algorithm', ['brute', 'kd_tree', 'ball_tree'])
def test_dbscan_core_samples_toy(algorithm):
    X = [[0], [2], [3], [4], [6], [8], [10]]
    n_samples = len(X)
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=1)
    assert_array_equal(core_samples, np.arange(n_samples))
    assert_array_equal(labels, [0, 1, 1, 1, 2, 3, 4])
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=2)
    assert_array_equal(core_samples, [1, 2, 3])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=3)
    assert_array_equal(core_samples, [2])
    assert_array_equal(labels, [-1, 0, 0, 0, -1, -1, -1])
    core_samples, labels = dbscan(X, algorithm=algorithm, eps=1, min_samples=4)
    assert_array_equal(core_samples, [])
    assert_array_equal(labels, np.full(n_samples, -1.0))