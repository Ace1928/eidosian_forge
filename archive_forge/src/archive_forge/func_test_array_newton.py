import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_array_newton(self):
    """test newton with array"""

    def f1(x, *a):
        b = a[0] + x * a[3]
        return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

    def f1_1(x, *a):
        b = a[3] / a[5]
        return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

    def f1_2(x, *a):
        b = a[3] / a[5]
        return -a[2] * np.exp(a[0] / a[5] + x * b) * b ** 2
    a0 = np.array([5.32725221, 5.48673747, 5.49539973, 5.36387202, 4.80237316, 1.43764452, 5.23063958, 5.46094772, 5.50512718, 5.4204629])
    a1 = (np.sin(range(10)) + 1.0) * 7.0
    args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
    x0 = [7.0] * 10
    x = zeros.newton(f1, x0, f1_1, args)
    x_expected = (6.17264965, 11.7702805, 12.2219954, 7.11017681, 1.18151293, 0.143707955, 4.31928228, 10.5419107, 12.755249, 8.91225749)
    assert_allclose(x, x_expected)
    x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
    assert_allclose(x, x_expected)
    x = zeros.newton(f1, x0, args=args)
    assert_allclose(x, x_expected)