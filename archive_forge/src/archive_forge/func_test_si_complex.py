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
def test_si_complex(self):

    def si(z):
        return sc.sici(z)[0]
    assert_mpmath_equal(si, mpmath.si, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-12)