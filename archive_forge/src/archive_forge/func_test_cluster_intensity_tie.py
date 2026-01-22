import warnings
import numpy as np
import pytest
from sklearn.cluster import MeanShift, estimate_bandwidth, get_bin_seeds, mean_shift
from sklearn.datasets import make_blobs
from sklearn.metrics import v_measure_score
from sklearn.utils._testing import assert_allclose, assert_array_equal
def test_cluster_intensity_tie(global_dtype):
    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=global_dtype)
    c1 = MeanShift(bandwidth=2).fit(X)
    X = np.array([[4, 7], [3, 5], [3, 6], [1, 1], [2, 1], [1, 0]], dtype=global_dtype)
    c2 = MeanShift(bandwidth=2).fit(X)
    assert_array_equal(c1.labels_, [1, 1, 1, 0, 0, 0])
    assert_array_equal(c2.labels_, [0, 0, 0, 1, 1, 1])