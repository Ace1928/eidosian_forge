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
def test_exprel(self):
    assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), [Arg(a=-np.log(np.finfo(np.float64).max), b=np.log(np.finfo(np.float64).max))])
    assert_mpmath_equal(sc.exprel, lambda x: mpmath.expm1(x) / x if x != 0 else mpmath.mpf('1.0'), np.array([1e-12, 1e-24, 0, 1000000000000.0, 1e+24, np.inf]), rtol=1e-11)
    assert_(np.isinf(sc.exprel(np.inf)))
    assert_(sc.exprel(-np.inf) == 0)