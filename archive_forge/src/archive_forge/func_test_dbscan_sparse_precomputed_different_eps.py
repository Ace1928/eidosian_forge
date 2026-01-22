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
def test_dbscan_sparse_precomputed_different_eps():
    lower_eps = 0.2
    nn = NearestNeighbors(radius=lower_eps).fit(X)
    D_sparse = nn.radius_neighbors_graph(X, mode='distance')
    dbscan_lower = dbscan(D_sparse, eps=lower_eps, metric='precomputed')
    higher_eps = lower_eps + 0.7
    nn = NearestNeighbors(radius=higher_eps).fit(X)
    D_sparse = nn.radius_neighbors_graph(X, mode='distance')
    dbscan_higher = dbscan(D_sparse, eps=lower_eps, metric='precomputed')
    assert_array_equal(dbscan_lower[0], dbscan_higher[0])
    assert_array_equal(dbscan_lower[1], dbscan_higher[1])