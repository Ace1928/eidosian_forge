import warnings
import numpy as np
import pytest
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.cluster._optics import _extend_region, _extract_xi_labels
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_minimum_number_of_sample_check():
    msg = 'min_samples must be no greater than'
    X = [[1, 1]]
    clust = OPTICS(max_eps=5.0 * 0.3, min_samples=10, min_cluster_size=1.0)
    with pytest.raises(ValueError, match=msg):
        clust.fit(X)