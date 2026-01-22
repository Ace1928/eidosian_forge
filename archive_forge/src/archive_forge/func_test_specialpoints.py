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
def test_specialpoints(self):
    assert_almost_equal(special.tandg(0), 0.0, 14)
    assert_almost_equal(special.tandg(45), 1.0, 14)
    assert_almost_equal(special.tandg(-45), -1.0, 14)
    assert_almost_equal(special.tandg(135), -1.0, 14)
    assert_almost_equal(special.tandg(-135), 1.0, 14)
    assert_almost_equal(special.tandg(180), 0.0, 14)
    assert_almost_equal(special.tandg(-180), 0.0, 14)
    assert_almost_equal(special.tandg(225), 1.0, 14)
    assert_almost_equal(special.tandg(-225), -1.0, 14)
    assert_almost_equal(special.tandg(315), -1.0, 14)
    assert_almost_equal(special.tandg(-315), 1.0, 14)