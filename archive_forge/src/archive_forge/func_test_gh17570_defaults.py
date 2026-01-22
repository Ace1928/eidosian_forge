import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh17570_defaults(self):
    res_newton_default = root_scalar(f1, method='newton', x0=3, xtol=1e-06)
    res_secant_default = root_scalar(f1, method='secant', x0=3, x1=2, xtol=1e-06)
    res_secant = newton(f1, x0=3, x1=2, tol=1e-06, full_output=True)[1]
    assert_allclose(f1(res_newton_default.root), 0, atol=1e-06)
    assert res_newton_default.root.shape == tuple()
    assert_allclose(f1(res_secant_default.root), 0, atol=1e-06)
    assert res_secant_default.root.shape == tuple()
    assert_allclose(f1(res_secant.root), 0, atol=1e-06)
    assert res_secant.root.shape == tuple()
    assert res_secant_default.root == res_secant.root != res_newton_default.iterations
    assert res_secant_default.iterations == res_secant_default.function_calls - 1 == res_secant.iterations != res_newton_default.iterations == res_newton_default.function_calls / 2