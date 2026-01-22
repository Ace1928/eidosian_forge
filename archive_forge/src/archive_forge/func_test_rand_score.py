import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_rand_score():
    clustering1 = [0, 0, 0, 1, 1, 1]
    clustering2 = [0, 1, 0, 1, 2, 2]
    D11 = 2 * 2
    D10 = 2 * 4
    D01 = 2 * 1
    D00 = 5 * 6 - D11 - D01 - D10
    expected_numerator = D00 + D11
    expected_denominator = D00 + D01 + D10 + D11
    expected = expected_numerator / expected_denominator
    assert_allclose(rand_score(clustering1, clustering2), expected)