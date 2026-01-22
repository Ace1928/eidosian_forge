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
def test_factorialx_negative_array(self, exact):
    assert_func = assert_array_equal if exact else assert_allclose
    assert_func(special.factorial([-5, -4, 0, 1], exact=exact), [0, 0, 1, 1])
    assert_func(special.factorial2([-5, -4, 0, 1], exact=exact), [0, 0, 1, 1])
    assert_func(special.factorialk([-5, -4, 0, 1], 3, exact=True), [0, 0, 1, 1])