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
def test_spherical_in(self):

    def mp_spherical_in(n, z):
        arg = mpmath.mpmathify(z)
        out = mpmath.besseli(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
        if arg.imag == 0:
            return out.real
        else:
            return out
    assert_mpmath_equal(lambda n, z: sc.spherical_in(int(n), z), exception_to_nan(mp_spherical_in), [IntArg(0, 200), Arg()], dps=200, atol=10 ** (-278))