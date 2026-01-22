import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy.special import log_ndtr, ndtri_exp
from scipy.special._testutils import assert_func_equal
@pytest.mark.parametrize('interval,expected_rtol', [((-10, -2), 1e-14), ((-2, -0.14542), 1e-12), ((-0.14542, -1e-06), 1e-10), ((-1e-06, 0), 1e-06)])
def test_in_interval(self, interval, expected_rtol, uniform_random_points):
    left, right = interval
    points = (right - left) * uniform_random_points + left
    assert_func_equal(log_ndtr_ndtri_exp, lambda y: y, points, rtol=expected_rtol, nan_ok=True)