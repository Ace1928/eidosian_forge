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
def test_spherical_kn(self):

    def mp_spherical_kn(n, z):
        out = mpmath.besselk(n + mpmath.mpf(1) / 2, z) * mpmath.sqrt(mpmath.pi / (2 * mpmath.mpmathify(z)))
        if mpmath.mpmathify(z).imag == 0:
            return out.real
        else:
            return out
    assert_mpmath_equal(lambda n, z: sc.spherical_kn(int(n), z), exception_to_nan(mp_spherical_kn), [IntArg(0, 150), Arg()], dps=100)