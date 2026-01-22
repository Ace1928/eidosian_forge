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
def test_improper_integrals(self):

    def f(x):
        return np.exp(-x ** 2)
    a = [-np.inf, 0, -np.inf, np.inf, -20, -np.inf, -20]
    b = [np.inf, np.inf, 0, -np.inf, 20, 20, np.inf]
    ref = np.sqrt(np.pi)
    res = _tanhsinh(f, a, b)
    assert_allclose(res.integral, [ref, ref / 2, ref / 2, -ref, ref, ref, ref])