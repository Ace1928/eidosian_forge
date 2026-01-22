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
def test_elliprg(self):
    assert_allclose(elliprg(1, 1, 1), 1)
    assert_allclose(elliprg(0, 0, 1), 0.5)
    assert_allclose(elliprg(0, 0, 0), 0)
    assert np.isinf(elliprg(1, inf, 1))
    assert np.isinf(elliprg(complex(inf), 1, 1))
    args = array([[0.0, 16.0, 16.0], [2.0, 3.0, 4.0], [0.0, 1j, -1j], [-1.0 + 1j, 1j, 0.0], [-1j, -1.0 + 1j, 1j], [0.0, 0.0796, 4.0]])
    expected_results = array([np.pi, 1.7255030280692, 0.42360654239699, 0.44660591677018 + 0.70768352357515j, 0.36023392184473 + 0.40348623401722j, 1.0284758090288])
    for i, arr in enumerate(args):
        assert_allclose(elliprg(*arr), expected_results[i])