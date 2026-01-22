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
def test_cospi(self):
    eps = np.finfo(float).eps
    assert_mpmath_equal(_cospi, mpmath.cospi, [Arg()], nan_ok=False, rtol=2 * eps)