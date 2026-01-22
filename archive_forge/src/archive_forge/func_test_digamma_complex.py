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
def test_digamma_complex(self):

    def param_filter(z):
        return np.where((z.real < 0) & (np.abs(z.imag) < 1.12), False, True)
    assert_mpmath_equal(sc.digamma, exception_to_nan(mpmath.digamma), [ComplexArg()], rtol=1e-13, dps=40, param_filter=param_filter)