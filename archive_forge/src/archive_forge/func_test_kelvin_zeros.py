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
def test_kelvin_zeros(self):
    tmp = special.kelvin_zeros(5)
    berz, beiz, kerz, keiz, berpz, beipz, kerpz, keipz = tmp
    assert_array_almost_equal(berz, array([2.84892, 7.23883, 11.67396, 16.11356, 20.55463]), 4)
    assert_array_almost_equal(beiz, array([5.02622, 9.45541, 13.89349, 18.33398, 22.77544]), 4)
    assert_array_almost_equal(kerz, array([1.71854, 6.12728, 10.56294, 15.00269, 19.44382]), 4)
    assert_array_almost_equal(keiz, array([3.91467, 8.34422, 12.78256, 17.22314, 21.66464]), 4)
    assert_array_almost_equal(berpz, array([6.03871, 10.51364, 14.96844, 19.41758, 23.8643]), 4)
    assert_array_almost_equal(beipz, array([3.77267, 8.28099, 12.74215, 17.19343, 21.64114]), 4)
    assert_array_almost_equal(kerpz, array([2.66584, 7.17212, 11.63218, 16.08312, 20.53068]), 4)
    assert_array_almost_equal(keipz, array([4.93181, 9.40405, 13.85827, 18.30717, 22.75379]), 4)