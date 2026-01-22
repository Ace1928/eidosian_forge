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
@pytest.mark.parametrize('n', range(30, 180, 11))
def test_factorial2_accuracy(self, n):
    rtol = 2e-14 if sys.platform == 'win32' else 1e-15
    assert_allclose(float(special.factorial2(n, exact=True)), special.factorial2(n, exact=False), rtol=rtol)
    assert_allclose(special.factorial2([n], exact=True).astype(float), special.factorial2([n], exact=False), rtol=rtol)