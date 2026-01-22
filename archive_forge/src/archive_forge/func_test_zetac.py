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
def test_zetac(self):
    assert_mpmath_equal(sc.zetac, lambda x: mpmath.zeta(x) - 1 if x != 1 else mpmath.inf, [Arg(-100, 100)], nan_ok=False, dps=45, rtol=5e-13)