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
def test_dbscan_precomputed_metric_with_degenerate_input_arrays():
    X = np.eye(10)
    labels = DBSCAN(eps=0.5, metric='precomputed').fit(X).labels_
    assert len(set(labels)) == 1
    X = np.zeros((10, 10))
    labels = DBSCAN(eps=0.5, metric='precomputed').fit(X).labels_
    assert len(set(labels)) == 1