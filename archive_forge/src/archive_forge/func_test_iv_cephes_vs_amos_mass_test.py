import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
@pytest.mark.slow
def test_iv_cephes_vs_amos_mass_test(self):
    N = 1000000
    np.random.seed(1)
    v = np.random.pareto(0.5, N) * (-1) ** np.random.randint(2, size=N)
    x = np.random.pareto(0.2, N) * (-1) ** np.random.randint(2, size=N)
    imsk = np.random.randint(8, size=N) == 0
    v[imsk] = v[imsk].astype(int)
    with np.errstate(all='ignore'):
        c1 = special.iv(v, x)
        c2 = special.iv(v, x + 0j)
        c1[abs(c1) > 1e+300] = np.inf
        c2[abs(c2) > 1e+300] = np.inf
        c1[abs(c1) < 1e-300] = 0
        c2[abs(c2) < 1e-300] = 0
        dc = abs(c1 / c2 - 1)
        dc[np.isnan(dc)] = 0
    k = np.argmax(dc)
    assert_(dc[k] < 2e-07, (v[k], x[k], special.iv(v[k], x[k]), special.iv(v[k], x[k] + 0j)))