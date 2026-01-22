import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_zero_der_nz_dp():
    """Test secant method with a non-zero dp, but an infinite newton step"""
    dx = np.finfo(float).eps ** 0.33
    p0 = (200.0 - dx) / (2.0 + dx)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'RMS of')
        x = zeros.newton(lambda y: (y - 100.0) ** 2, x0=[p0] * 10)
    assert_allclose(x, [100] * 10)
    p0 = (2.0 - 0.0001) / (2.0 + 0.0001)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Tolerance of')
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, 1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y - 1.0) ** 2, x0=p0, disp=True)
    p0 = (-2.0 + 0.0001) / (2.0 + 0.0001)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Tolerance of')
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=False)
    assert_allclose(x, -1)
    with pytest.raises(RuntimeError, match='Tolerance of'):
        x = zeros.newton(lambda y: (y + 1.0) ** 2, x0=p0, disp=True)