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
def test_jumpstart(self):
    a, b = (-np.inf, np.inf)

    def f(x):
        return np.exp(-x * x)

    def callback(res):
        callback.integrals.append(res.integral)
        callback.errors.append(res.error)
    callback.integrals = []
    callback.errors = []
    maxlevel = 4
    _tanhsinh(f, a, b, minlevel=0, maxlevel=maxlevel, callback=callback)
    integrals = []
    errors = []
    for i in range(maxlevel + 1):
        res = _tanhsinh(f, a, b, minlevel=i, maxlevel=i)
        integrals.append(res.integral)
        errors.append(res.error)
    assert_allclose(callback.integrals[1:], integrals, rtol=1e-15)
    assert_allclose(callback.errors[1:], errors, rtol=1e-15, atol=1e-16)