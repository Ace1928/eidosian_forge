import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_adjustment_for_chance():
    n_clusters_range = [2, 10, 50, 90]
    n_samples = 100
    n_runs = 10
    scores = uniform_labelings_scores(adjusted_rand_score, n_samples, n_clusters_range, n_runs)
    max_abs_scores = np.abs(scores).max(axis=1)
    assert_array_almost_equal(max_abs_scores, [0.02, 0.03, 0.03, 0.02], 2)