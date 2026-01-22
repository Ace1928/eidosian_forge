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
@pytest.mark.parametrize('k', list(range(1, 5)) + [10, 20])
@pytest.mark.parametrize('n', list(range(0, 22)) + list(range(22, 100, 11)))
def test_factorialk_int_reference(self, n, k):
    correct = functools.reduce(operator.mul, list(range(n, 0, -k)), 1)
    assert_array_equal(correct, special.factorialk(n, k, True))
    assert_array_equal(correct, special.factorialk([n], k, True)[0])