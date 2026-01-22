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
def test_spence_complex(self):

    def dilog(z):
        return mpmath.polylog(2, 1 - z)
    assert_mpmath_equal(sc.spence, exception_to_nan(dilog), [ComplexArg()], rtol=1e-14)