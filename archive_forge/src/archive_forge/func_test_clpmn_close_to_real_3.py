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
def test_clpmn_close_to_real_3(self):
    eps = 1e-10
    m = 1
    n = 3
    x = 0.5
    clp_plus = special.clpmn(m, n, x + 1j * eps, 3)[0][m, n]
    clp_minus = special.clpmn(m, n, x - 1j * eps, 3)[0][m, n]
    assert_array_almost_equal(array([clp_plus, clp_minus]), array([special.lpmv(m, n, x) * np.exp(-0.5j * m * np.pi), special.lpmv(m, n, x) * np.exp(0.5j * m * np.pi)]), 7)