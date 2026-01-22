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
@pytest.mark.parametrize('a, b, y, ref', [(14.208308325339239, 14.208308325339239, 7.703145458496392e-307, 8.566004561846704e-23), (14.0, 14.5, 1e-280, 2.9343915006642424e-21), (3.5, 15.0, 4e-95, 1.3290751429289227e-28), (10.0, 1.25, 2e-234, 3.982659092143654e-24), (4.0, 99997.0, 5e-88, 3.309800566862242e-27)])
def test_betaincinv_tiny_y(self, a, b, y, ref):
    x = special.betaincinv(a, b, y)
    assert_allclose(x, ref, rtol=1e-14)