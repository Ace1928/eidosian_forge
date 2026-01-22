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
def test_jacobi(self):
    a = 5 * np.random.random() - 1
    b = 5 * np.random.random() - 1
    P0 = special.jacobi(0, a, b)
    P1 = special.jacobi(1, a, b)
    P2 = special.jacobi(2, a, b)
    P3 = special.jacobi(3, a, b)
    assert_array_almost_equal(P0.c, [1], 13)
    assert_array_almost_equal(P1.c, array([a + b + 2, a - b]) / 2.0, 13)
    cp = [(a + b + 3) * (a + b + 4), 4 * (a + b + 3) * (a + 2), 4 * (a + 1) * (a + 2)]
    p2c = [cp[0], cp[1] - 2 * cp[0], cp[2] - cp[1] + cp[0]]
    assert_array_almost_equal(P2.c, array(p2c) / 8.0, 13)
    cp = [(a + b + 4) * (a + b + 5) * (a + b + 6), 6 * (a + b + 4) * (a + b + 5) * (a + 3), 12 * (a + b + 4) * (a + 2) * (a + 3), 8 * (a + 1) * (a + 2) * (a + 3)]
    p3c = [cp[0], cp[1] - 3 * cp[0], cp[2] - 2 * cp[1] + 3 * cp[0], cp[3] - cp[2] + cp[1] - cp[0]]
    assert_array_almost_equal(P3.c, array(p3c) / 48.0, 13)