import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_v_measure_and_mutual_information(seed=36):
    for i in np.logspace(1, 4, 4).astype(int):
        random_state = np.random.RandomState(seed)
        labels_a, labels_b = (random_state.randint(0, 10, i), random_state.randint(0, 10, i))
        assert_almost_equal(v_measure_score(labels_a, labels_b), 2.0 * mutual_info_score(labels_a, labels_b) / (entropy(labels_a) + entropy(labels_b)), 0)
        avg = 'arithmetic'
        assert_almost_equal(v_measure_score(labels_a, labels_b), normalized_mutual_info_score(labels_a, labels_b, average_method=avg))