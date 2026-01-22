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
def test_spherical_jn(self):

    def mp_spherical_jn(n, z):
        arg = mpmath.mpmathify(z)
        out = mpmath.besselj(n + mpmath.mpf(1) / 2, arg) / mpmath.sqrt(2 * arg / mpmath.pi)
        if arg.imag == 0:
            return out.real
        else:
            return out
    assert_mpmath_equal(lambda n, z: sc.spherical_jn(int(n), z), exception_to_nan(mp_spherical_jn), [IntArg(0, 200), Arg(-100000000.0, 100000000.0)], dps=300)