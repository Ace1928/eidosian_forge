import numpy as np
import pytest
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('bisecting_strategy', ['biggest_inertia', 'largest_cluster'])
@pytest.mark.parametrize('init', ['k-means++', 'random'])
def test_three_clusters(bisecting_strategy, init):
    """Tries to perform bisect k-means for three clusters to check
    if splitting data is performed correctly.
    """
    X = np.array([[1, 1], [10, 1], [3, 1], [10, 0], [2, 1], [10, 2], [10, 8], [10, 9], [10, 10]])
    bisect_means = BisectingKMeans(n_clusters=3, random_state=0, bisecting_strategy=bisecting_strategy, init=init)
    bisect_means.fit(X)
    expected_centers = [[2, 1], [10, 1], [10, 9]]
    expected_labels = [0, 1, 0, 1, 0, 1, 2, 2, 2]
    assert_allclose(sorted(expected_centers), sorted(bisect_means.cluster_centers_.tolist()))
    assert_allclose(v_measure_score(expected_labels, bisect_means.labels_), 1.0)