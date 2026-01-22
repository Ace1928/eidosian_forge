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
def test_quadrature_single_args(self):

    def myfunc(x, n):
        return 1e+90 * cos(n * x - 1.8 * sin(x)) / pi
    val, err = quadrature(myfunc, 0, pi, args=2, rtol=1e-10)
    table_val = 1e+90 * 0.30614353532540295
    assert_allclose(val, table_val, rtol=1e-10)