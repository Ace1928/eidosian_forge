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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dbscan_no_core_samples(csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < 0.8] = 0
    for X_ in [X, csr_container(X)]:
        db = DBSCAN(min_samples=6).fit(X_)
        assert_array_equal(db.components_, np.empty((0, X_.shape[1])))
        assert_array_equal(db.labels_, -1)
        assert db.core_sample_indices_.shape == (0,)