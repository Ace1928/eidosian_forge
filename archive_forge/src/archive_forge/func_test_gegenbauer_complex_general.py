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
@nonfunctional_tooslow
def test_gegenbauer_complex_general(self):
    assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(n.real, a.real, x), exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), ComplexArg()])