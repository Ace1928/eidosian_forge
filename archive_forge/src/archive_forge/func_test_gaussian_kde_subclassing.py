from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_gaussian_kde_subclassing():
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)
    kde = stats.gaussian_kde(x1)
    ys = kde(xs)
    kde1 = _kde_subclass1(x1)
    y1 = kde1(xs)
    assert_array_almost_equal_nulp(ys, y1, nulp=10)
    kde2 = _kde_subclass2(x1)
    y2 = kde2(xs)
    assert_array_almost_equal_nulp(ys, y2, nulp=10)
    kde4 = _kde_subclass4(x1)
    y4 = kde4(x1)
    y_expected = [0.06292987, 0.06346938, 0.05860291, 0.08657652, 0.07904017]
    assert_array_almost_equal(y_expected, y4, decimal=6)
    kde5 = kde
    kde5.covariance_factor = lambda: kde.factor
    kde5._compute_covariance()
    y5 = kde5(xs)
    assert_array_almost_equal_nulp(ys, y5, nulp=10)