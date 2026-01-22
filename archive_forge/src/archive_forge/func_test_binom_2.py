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
def test_binom_2(self):
    np.random.seed(1234)
    n = np.r_[np.logspace(1, 300, 20)]
    k = np.arange(0, 102)
    nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
    assert_func_equal(cephes.binom, cephes.binom(nk[:, 0], nk[:, 1] * (1 + 1e-15)), nk, atol=1e-10, rtol=1e-10)