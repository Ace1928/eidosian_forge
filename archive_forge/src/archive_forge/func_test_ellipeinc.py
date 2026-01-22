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
def test_ellipeinc(self):
    eleinc = special.ellipeinc(pi / 2, 0.2)
    ele = special.ellipe(0.2)
    assert_almost_equal(eleinc, ele, 14)
    alpha, phi = (52 * pi / 180, 35 * pi / 180)
    m = sin(alpha) ** 2
    eleinc = special.ellipeinc(phi, m)
    assert_almost_equal(eleinc, 0.58823065, 8)
    assert_equal(special.ellipeinc(pi / 2, 0.0), pi / 2)
    assert_equal(special.ellipeinc(pi / 2, 1.0), 1.0)
    assert_equal(special.ellipeinc(pi / 2, -np.inf), np.inf)
    assert_equal(special.ellipeinc(pi / 2, np.nan), np.nan)
    assert_equal(special.ellipeinc(pi / 2, 2), np.nan)
    assert_equal(special.ellipeinc(0, 0.5), 0.0)
    assert_equal(special.ellipeinc(np.inf, 0.5), np.inf)
    assert_equal(special.ellipeinc(-np.inf, 0.5), -np.inf)
    assert_equal(special.ellipeinc(np.inf, -np.inf), np.inf)
    assert_equal(special.ellipeinc(-np.inf, -np.inf), -np.inf)
    assert_equal(special.ellipeinc(np.inf, np.inf), np.nan)
    assert_equal(special.ellipeinc(-np.inf, np.inf), np.nan)
    assert_equal(special.ellipeinc(np.nan, 0.5), np.nan)
    assert_equal(special.ellipeinc(np.nan, np.nan), np.nan)
    assert_allclose(special.ellipeinc(1.5707, -10), 3.6388185585822876)