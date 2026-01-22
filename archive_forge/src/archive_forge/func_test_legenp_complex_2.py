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
def test_legenp_complex_2(self):

    def clpnm(n, m, z):
        try:
            return sc.clpmn(m.real, n.real, z, type=2)[0][-1, -1]
        except ValueError:
            return np.nan

    def legenp(n, m, z):
        if abs(z) < 1e-15:
            return np.nan
        return exception_to_nan(mpmath.legenp)(int(n.real), int(m.real), z, type=2)
    x = np.array([-2, -0.99, -0.5, 0, 1e-05, 0.5, 0.99, 20, 2000.0])
    y = np.array([-1000.0, -0.5, 0.5, 1.3])
    z = (x[:, None] + 1j * y[None, :]).ravel()
    assert_mpmath_equal(clpnm, legenp, [FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg([-2, -1, 0, 1, 2, 10]), FixedArg(z)], rtol=1e-06, n=500)