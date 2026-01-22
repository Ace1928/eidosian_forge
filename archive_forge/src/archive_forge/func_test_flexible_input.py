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
def test_flexible_input(self):

    def func(x):
        return stats.norm.pdf(x, scale=2)
    res = qmc_quad(func, 0, 1)
    ref = stats.norm.cdf(1, scale=2) - stats.norm.cdf(0, scale=2)
    assert_allclose(res.integral, ref, 0.01)