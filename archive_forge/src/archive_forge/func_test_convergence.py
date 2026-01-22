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
def test_convergence(self):
    f = self.f1
    last_logerr = 0
    for i in range(4):
        res = _tanhsinh(f, 0, f.b, minlevel=0, maxlevel=i)
        logerr = self.error(res.integral, f.ref, log=True)
        assert logerr < last_logerr * 2 or logerr < -15.5
        last_logerr = logerr