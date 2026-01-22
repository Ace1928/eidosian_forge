import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
def test_beta_parameter():
    beta_test = 0.2
    h_test = 0.67
    c_test = 0.42
    v_test = (1 + beta_test) * h_test * c_test / (beta_test * h_test + c_test)
    h, c, v = homogeneity_completeness_v_measure([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2], beta=beta_test)
    assert_almost_equal(h, h_test, 2)
    assert_almost_equal(c, c_test, 2)
    assert_almost_equal(v, v_test, 2)
    v = v_measure_score([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 2], beta=beta_test)
    assert_almost_equal(v, v_test, 2)