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
def test_yn_zeros(self):
    an = special.yn_zeros(4, 2)
    assert_array_almost_equal(an, array([5.64515, 9.36162]), 5)
    an = special.yn_zeros(443, 5)
    assert_allclose(an, [450.1357309157809, 463.05692376675, 472.80651546418665, 481.27353184725627, 488.98055964441374], rtol=1e-15)