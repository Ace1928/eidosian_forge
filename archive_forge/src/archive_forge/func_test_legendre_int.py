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
def test_legendre_int(self):
    assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), Arg()], n=20000)
    assert_mpmath_equal(lambda n, x: sc.eval_legendre(int(n), x), lambda n, x: exception_to_nan(mpmath.legendre)(n, x, **HYPERKW), [IntArg(), FixedArg(np.logspace(-30, -4, 20))])