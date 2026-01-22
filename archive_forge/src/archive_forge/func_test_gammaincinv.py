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
@with_special_errors
def test_gammaincinv(self):
    y = special.gammaincinv(0.4, 0.4)
    x = special.gammainc(0.4, y)
    assert_almost_equal(x, 0.4, 1)
    y = special.gammainc(10, 0.05)
    x = special.gammaincinv(10, 2.5715803516000737e-20)
    assert_almost_equal(0.05, x, decimal=10)
    assert_almost_equal(y, 2.5715803516000737e-20, decimal=10)
    x = special.gammaincinv(50, 8.207547773884713e-18)
    assert_almost_equal(11.0, x, decimal=10)