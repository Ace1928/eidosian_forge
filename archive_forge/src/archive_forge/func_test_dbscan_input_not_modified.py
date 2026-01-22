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
@pytest.mark.parametrize('metric', ['precomputed', 'minkowski'])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS + [None])
def test_dbscan_input_not_modified(metric, csr_container):
    X = np.random.RandomState(0).rand(10, 10)
    X = csr_container(X) if csr_container is not None else X
    X_copy = X.copy()
    dbscan(X, metric=metric)
    if csr_container is not None:
        assert_array_equal(X.toarray(), X_copy.toarray())
    else:
        assert_array_equal(X, X_copy)