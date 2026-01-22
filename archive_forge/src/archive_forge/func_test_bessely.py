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
def test_bessely(self):

    def mpbessely(v, x):
        r = float(mpmath.bessely(v, x, **HYPERKW))
        if abs(r) > 1e+305:
            r = np.inf * np.sign(r)
        if abs(r) == 0 and x == 0:
            return np.nan
        return r
    assert_mpmath_equal(sc.yv, exception_to_nan(mpbessely), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], n=5000)