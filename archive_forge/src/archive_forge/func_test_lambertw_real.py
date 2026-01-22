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
@pytest.mark.xfail_on_32bit('see gh-3551 for bad points')
def test_lambertw_real(self):
    assert_mpmath_equal(lambda x, k: sc.lambertw(x, int(k.real)), lambda x, k: mpmath.lambertw(x, int(k.real)), [ComplexArg(-np.inf, np.inf), IntArg(0, 10)], rtol=1e-13, nan_ok=False)