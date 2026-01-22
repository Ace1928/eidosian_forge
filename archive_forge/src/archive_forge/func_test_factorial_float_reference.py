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
@pytest.mark.parametrize('exact', [True, False])
def test_factorial_float_reference(self, exact):

    def _check(n, expected):
        assert_allclose(special.factorial(n, exact=exact), expected)
        assert_allclose(special.factorial([n])[0], expected)
    _check(0.01, 0.9943258511915061)
    _check(1.11, 1.051609009483625)
    _check(5.55, 314.9503192327208)
    _check(11.1, 50983227.84411616)
    _check(33.3, 2.4933633396420364e+37)
    _check(55.5, 9.479934358436729e+73)
    _check(77.7, 3.060540559059579e+114)
    _check(99.9, 5.885840419492872e+157)
    _check(170.6243, 1.7969818574957104e+308)