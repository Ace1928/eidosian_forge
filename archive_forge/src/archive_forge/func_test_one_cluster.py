import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_cluster():
    """Test single cluster."""
    X = np.array([[1, 2], [10, 2], [10, 8]])
    bisect_means = BisectingKMeans(n_clusters=1, random_state=0).fit(X)
    assert all(bisect_means.labels_ == 0)
    assert all(bisect_means.predict(X) == 0)
    assert_allclose(bisect_means.cluster_centers_, X.mean(axis=0).reshape(1, -1))