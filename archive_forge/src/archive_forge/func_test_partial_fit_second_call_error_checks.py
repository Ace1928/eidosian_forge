import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.cluster.tests.common import generate_clustered_data
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin, v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_partial_fit_second_call_error_checks():
    X, y = make_blobs(n_samples=100)
    brc = Birch(n_clusters=3)
    brc.partial_fit(X, y)
    msg = 'X has 1 features, but Birch is expecting 2 features'
    with pytest.raises(ValueError, match=msg):
        brc.partial_fit(X[:, [0]], y)