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
def test_jnjnp_zeros(self):
    jn = special.jn

    def jnp(n, x):
        return (jn(n - 1, x) - jn(n + 1, x)) / 2
    for nt in range(1, 30):
        z, n, m, t = special.jnjnp_zeros(nt)
        for zz, nn, tt in zip(z, n, t):
            if tt == 0:
                assert_allclose(jn(nn, zz), 0, atol=1e-06)
            elif tt == 1:
                assert_allclose(jnp(nn, zz), 0, atol=1e-06)
            else:
                raise AssertionError('Invalid t return for nt=%d' % nt)