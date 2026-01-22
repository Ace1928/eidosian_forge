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
@pytest.mark.xfail(run=False, reason='apparently picks wrong function at |z| > 1')
def test_legenq(self):

    def lqnm(n, m, z):
        return sc.lqmn(m, n, z)[0][-1, -1]

    def legenq(n, m, z):
        if abs(z) < 1e-15:
            return np.nan
        return exception_to_nan(mpmath.legenq)(n, m, z, type=2)
    assert_mpmath_equal(lqnm, legenq, [IntArg(0, 100), IntArg(0, 100), Arg()])