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
def test_vs_series(self):
    """Check Struve function versus its power series"""
    for v in [-20, -10, -7.99, -3.4, -1, 0, 1, 3.4, 12.49, 16]:
        for z in [1, 10, 19, 21, 30]:
            value, err = self._series(v, z)
            (assert_allclose(special.struve(v, z), value, rtol=0, atol=err), (v, z))