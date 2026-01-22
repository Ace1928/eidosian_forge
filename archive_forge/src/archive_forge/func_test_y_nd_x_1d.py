import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
def test_y_nd_x_1d(self):
    y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
    x = np.arange(4) ** 2
    ys_expected = (np.array([[[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]], [[40.0, 44.0, 48.0, 52.0], [56.0, 60.0, 64.0, 68.0]]]), np.array([[[2.0, 3.0, 4.0, 5.0]], [[10.0, 11.0, 12.0, 13.0]], [[18.0, 19.0, 20.0, 21.0]]]), np.array([[[0.5, 5.0, 17.5], [4.5, 21.0, 53.5]], [[8.5, 37.0, 89.5], [12.5, 53.0, 125.5]], [[16.5, 69.0, 161.5], [20.5, 85.0, 197.5]]]))
    for axis, y_expected in zip([0, 1, 2], ys_expected):
        y_int = cumulative_trapezoid(y, x=x[:y.shape[axis]], axis=axis, initial=None)
        assert_allclose(y_int, y_expected)