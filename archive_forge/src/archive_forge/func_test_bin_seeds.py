import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_bin_seeds(global_dtype):
    X = np.array([[1.0, 1.0], [1.4, 1.4], [1.8, 1.2], [2.0, 1.0], [2.1, 1.1], [0.0, 0.0]], dtype=global_dtype)
    ground_truth = {(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)}
    test_bins = get_bin_seeds(X, 1, 1)
    test_result = set((tuple(p) for p in test_bins))
    assert len(ground_truth.symmetric_difference(test_result)) == 0
    ground_truth = {(1.0, 1.0), (2.0, 1.0)}
    test_bins = get_bin_seeds(X, 1, 2)
    test_result = set((tuple(p) for p in test_bins))
    assert len(ground_truth.symmetric_difference(test_result)) == 0
    with warnings.catch_warnings(record=True):
        test_bins = get_bin_seeds(X, 0.01, 1)
    assert_allclose(test_bins, X)
    X, _ = make_blobs(n_samples=100, n_features=2, centers=[[0, 0], [1, 1]], cluster_std=0.1, random_state=0)
    X = X.astype(global_dtype, copy=False)
    test_bins = get_bin_seeds(X, 1)
    assert_array_equal(test_bins, [[0, 0], [1, 1]])