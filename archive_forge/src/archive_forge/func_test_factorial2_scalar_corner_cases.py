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
@pytest.mark.parametrize('exact', [True, False])
@pytest.mark.parametrize('n', [1, 1.1, 2 + 2j, np.nan, None], ids=['1', '1.1', '2+2j', 'NaN', 'None'])
def test_factorial2_scalar_corner_cases(self, n, exact):
    if n is None or n is np.nan or np.issubdtype(type(n), np.integer):
        result = special.factorial2(n, exact=exact)
        exp = np.nan if n is np.nan or n is None else special.factorial(n)
        assert_equal(result, exp)
    else:
        with pytest.raises(ValueError, match='factorial2 does not*'):
            special.factorial2(n, exact=exact)