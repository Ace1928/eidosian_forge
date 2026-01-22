import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh9551_raise_error_if_disp_true():
    """Test that if disp is true then zero derivative raises RuntimeError"""

    def f(x):
        return x * x + 1

    def f_p(x):
        return 2 * x
    assert_warns(RuntimeWarning, zeros.newton, f, 1.0, f_p, disp=False)
    with pytest.raises(RuntimeError, match='^Derivative was zero\\. Failed to converge after \\d+ iterations, value is [+-]?\\d*\\.\\d+\\.$'):
        zeros.newton(f, 1.0, f_p)
    root = zeros.newton(f, complex(10.0, 10.0), f_p)
    assert_allclose(root, complex(0.0, 1.0))