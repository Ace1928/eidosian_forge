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
@pytest.mark.parametrize('a, b, x, p', [(2, 4, 0.3138101704556974, 0.5), (0.0342, 171.0, 1e-10, 0.5526991690180709), (0.0342, 171, 8.42313169354797e-21, 0.25), (0.0002742794749792665, 289206.03125, 1.639984034231756e-56, 0.9688708782196045), (4, 99997, 0.0001947841578892121, 0.999995)])
def test_betainc_betaincinv(self, a, b, x, p):
    p1 = special.betainc(a, b, x)
    assert_allclose(p1, p, rtol=1e-15)
    x1 = special.betaincinv(a, b, p)
    assert_allclose(x1, x, rtol=5e-13)