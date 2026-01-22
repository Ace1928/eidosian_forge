from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_minimize_quadratic_1d(self):
    a = 5
    b = -1
    t, y = minimize_quadratic_1d(a, b, 1, 2)
    assert_equal(t, 1)
    assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
    t, y = minimize_quadratic_1d(a, b, -2, -1)
    assert_equal(t, -1)
    assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
    t, y = minimize_quadratic_1d(a, b, -1, 1)
    assert_equal(t, 0.1)
    assert_allclose(y, a * t ** 2 + b * t, rtol=1e-15)
    c = 10
    t, y = minimize_quadratic_1d(a, b, -1, 1, c=c)
    assert_equal(t, 0.1)
    assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
    t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf, c=c)
    assert_equal(t, 0.1)
    assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
    t, y = minimize_quadratic_1d(a, b, 0, np.inf, c=c)
    assert_equal(t, 0.1)
    assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
    t, y = minimize_quadratic_1d(a, b, -np.inf, 0, c=c)
    assert_equal(t, 0)
    assert_allclose(y, a * t ** 2 + b * t + c, rtol=1e-15)
    a = -1
    b = 0.2
    t, y = minimize_quadratic_1d(a, b, -np.inf, np.inf)
    assert_equal(y, -np.inf)
    t, y = minimize_quadratic_1d(a, b, 0, np.inf)
    assert_equal(t, np.inf)
    assert_equal(y, -np.inf)
    t, y = minimize_quadratic_1d(a, b, -np.inf, 0)
    assert_equal(t, -np.inf)
    assert_equal(y, -np.inf)