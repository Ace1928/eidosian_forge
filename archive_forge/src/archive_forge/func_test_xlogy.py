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
def test_xlogy():

    def xfunc(x, y):
        with np.errstate(invalid='ignore'):
            if x == 0 and (not np.isnan(y)):
                return x
            else:
                return x * np.log(y)
    z1 = np.asarray([(0, 0), (0, np.nan), (0, np.inf), (1.0, 2.0)], dtype=float)
    z2 = np.r_[z1, [(0, 1j), (1, 1j)]]
    w1 = np.vectorize(xfunc)(z1[:, 0], z1[:, 1])
    assert_func_equal(special.xlogy, w1, z1, rtol=1e-13, atol=1e-13)
    w2 = np.vectorize(xfunc)(z2[:, 0], z2[:, 1])
    assert_func_equal(special.xlogy, w2, z2, rtol=1e-13, atol=1e-13)