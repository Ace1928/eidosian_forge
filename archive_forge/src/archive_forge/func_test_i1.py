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
def test_i1(self):
    values = [[0.0, 0.0], [1e-10, 4.9999999995e-11], [0.1, 0.0452984468], [0.5, 0.1564208032], [1.0, 0.2079104154], [5.0, 0.1639722669], [20.0, 0.0875062222]]
    for i, (x, v) in enumerate(values):
        cv = special.i1(x) * exp(-x)
        assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)