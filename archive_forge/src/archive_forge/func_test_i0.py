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
def test_i0(self):
    values = [[0.0, 1.0], [1e-10, 1.0], [0.1, 0.9071009258], [0.5, 0.6450352706], [1.0, 0.4657596077], [2.5, 0.2700464416], [5.0, 0.1835408126], [20.0, 0.0897803119]]
    for i, (x, v) in enumerate(values):
        cv = special.i0(x) * exp(-x)
        assert_almost_equal(cv, v, 8, err_msg='test #%d' % i)