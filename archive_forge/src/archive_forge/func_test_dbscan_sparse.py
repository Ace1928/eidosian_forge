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
@pytest.mark.parametrize('lil_container', LIL_CONTAINERS)
def test_dbscan_sparse(lil_container):
    core_sparse, labels_sparse = dbscan(lil_container(X), eps=0.8, min_samples=10)
    core_dense, labels_dense = dbscan(X, eps=0.8, min_samples=10)
    assert_array_equal(core_dense, core_sparse)
    assert_array_equal(labels_dense, labels_sparse)