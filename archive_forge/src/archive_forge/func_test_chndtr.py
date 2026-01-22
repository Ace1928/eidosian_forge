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
def test_chndtr(self):
    assert_equal(cephes.chndtr(0, 1, 0), 0.0)
    values = np.array([[25.0, 20.0, 400, 4.12106551123962e-57], [25.0, 8.0, 250, 2.3988026526832426e-29], [0.001, 8.0, 40.0, 5.376180620136604e-24], [0.01, 8.0, 40.0, 5.453962310559995e-20], [20.0, 2.0, 107, 1.393907435558196e-09], [22.5, 2.0, 107, 7.118033071381059e-09], [25.0, 2.0, 107, 3.110412448298649e-08], [3.0, 2.0, 1.0, 0.6206436532195436], [350.0, 300.0, 10.0, 0.9388012800627641], [100.0, 13.5, 10.0, 0.9999999965010421], [700.0, 20.0, 400, 0.9999999992568065], [150.0, 13.5, 10.0, 0.9999999999999998], [160.0, 13.5, 10.0, 1.0]])
    cdf = cephes.chndtr(values[:, 0], values[:, 1], values[:, 2])
    assert_allclose(cdf, values[:, 3], rtol=1e-12)
    assert_almost_equal(cephes.chndtr(np.inf, np.inf, 0), 2.0)
    assert_almost_equal(cephes.chndtr(2, 1, np.inf), 0.0)
    assert_(np.isnan(cephes.chndtr(np.nan, 1, 2)))
    assert_(np.isnan(cephes.chndtr(5, np.nan, 2)))
    assert_(np.isnan(cephes.chndtr(5, 1, np.nan)))