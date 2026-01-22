import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', ['secant', 'newton'])
def test_int_x0_gh19280(self, method):

    def f(x):
        return x ** (-2) - 2
    res = optimize.root_scalar(f, x0=1, method=method)
    assert res.converged
    assert_allclose(abs(res.root), 2 ** (-0.5))
    assert res.root.dtype == np.dtype(np.float64)