import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_pair_confusion_matrix_fully_dispersed():
    N = 100
    clustering1 = list(range(N))
    clustering2 = clustering1
    expected = np.array([[N * (N - 1), 0], [0, 0]])
    assert_array_equal(pair_confusion_matrix(clustering1, clustering2), expected)