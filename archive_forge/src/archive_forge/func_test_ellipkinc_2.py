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
def test_ellipkinc_2(self):
    mbad = 0.6835937500000001
    phi = 0.9272952180016123
    m = np.nextafter(mbad, 0)
    mvals = []
    for j in range(10):
        mvals.append(m)
        m = np.nextafter(m, 1)
    f = special.ellipkinc(phi, mvals)
    assert_array_almost_equal_nulp(f, np.full_like(f, 1.0259330100195334), 1)
    f1 = special.ellipkinc(phi + pi, mvals)
    assert_array_almost_equal_nulp(f1, np.full_like(f1, 5.1296650500976675), 2)