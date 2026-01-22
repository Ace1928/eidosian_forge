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
@pytest.mark.parametrize('is_exact, comp, kwargs', [(True, assert_equal, {}), (False, assert_allclose, {'rtol': 1e-12})])
def test_negative_integer(self, is_exact, comp, kwargs):
    comp(stirling2(-1, -1, exact=is_exact), 0, **kwargs)
    comp(stirling2(-1, 2, exact=is_exact), 0, **kwargs)
    comp(stirling2(2, -1, exact=is_exact), 0, **kwargs)