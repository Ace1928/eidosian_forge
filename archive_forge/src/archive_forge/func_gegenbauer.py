import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def gegenbauer(n, a, x):
    if abs(a) > 1e+100:
        return np.nan
    if n == 0:
        r = 1.0
    elif n == 1:
        r = 2 * a * x
    else:
        r = mpmath.gegenbauer(n, a, x)
    if float(r) == 0 and a < -1 and (float(a) == int(float(a))):
        r = mpmath.gegenbauer(n, a + mpmath.mpf('1e-50'), x)
        if abs(r) < mpmath.mpf('1e-50'):
            r = mpmath.mpf('0.0')
    if abs(r) > 1e+270:
        return np.inf
    return r