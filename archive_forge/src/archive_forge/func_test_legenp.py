import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def test_legenp(self):

    def lpnm(n, m, z):
        try:
            v = sc.lpmn(m, n, z)[0][-1, -1]
        except ValueError:
            return np.nan
        if abs(v) > 1e+306:
            v = np.inf * np.sign(v.real)
        return v

    def lpnm_2(n, m, z):
        v = sc.lpmv(m, n, z)
        if abs(v) > 1e+306:
            v = np.inf * np.sign(v.real)
        return v

    def legenp(n, m, z):
        if (z == 1 or z == -1) and int(n) == n:
            if m == 0:
                if n < 0:
                    n = -n - 1
                return mpmath.power(mpmath.sign(z), n)
            else:
                return 0
        if abs(z) < 1e-15:
            return np.nan
        typ = 2 if abs(z) < 1 else 3
        v = exception_to_nan(mpmath.legenp)(n, m, z, type=typ)
        if abs(v) > 1e+306:
            v = mpmath.inf * mpmath.sign(v.real)
        return v
    assert_mpmath_equal(lpnm, legenp, [IntArg(-100, 100), IntArg(-100, 100), Arg()])
    assert_mpmath_equal(lpnm_2, legenp, [IntArg(-100, 100), Arg(-100, 100), Arg(-1, 1)], atol=1e-10)