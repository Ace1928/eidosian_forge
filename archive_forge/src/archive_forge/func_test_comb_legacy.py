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
@pytest.mark.parametrize('repetition', [True, False])
@pytest.mark.parametrize('legacy', [True, False, _NoValue])
@pytest.mark.parametrize('k', [3.5, 3])
@pytest.mark.parametrize('N', [4.5, 4])
def test_comb_legacy(self, N, k, legacy, repetition):
    if legacy is not _NoValue:
        with pytest.warns(DeprecationWarning, match="Using 'legacy' keyword is deprecated"):
            result = special.comb(N, k, exact=True, legacy=legacy, repetition=repetition)
    else:
        result = special.comb(N, k, exact=True, legacy=legacy, repetition=repetition)
    if legacy:
        if repetition:
            N, k = (int(N + k - 1), int(k))
            repetition = False
        else:
            N, k = (int(N), int(k))
    with suppress_warnings() as sup:
        if legacy is not _NoValue:
            sup.filter(DeprecationWarning)
        expected = special.comb(N, k, legacy=legacy, repetition=repetition)
    assert_equal(result, expected)