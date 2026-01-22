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
def test_mathieu_modcem2(self):
    cephes.mathieu_modcem2(1, 1, 1)
    m = np.arange(0, 4)[:, None, None]
    q = np.r_[np.logspace(-2, 2, 10)][None, :, None]
    z = np.linspace(0, 1, 7)[None, None, :]
    y1 = cephes.mathieu_modcem2(m, q, -z)[0]
    fr = -cephes.mathieu_modcem2(m, q, 0)[0] / cephes.mathieu_modcem1(m, q, 0)[0]
    y2 = -cephes.mathieu_modcem2(m, q, z)[0] - 2 * fr * cephes.mathieu_modcem1(m, q, z)[0]
    assert_allclose(y1, y2, rtol=1e-10)