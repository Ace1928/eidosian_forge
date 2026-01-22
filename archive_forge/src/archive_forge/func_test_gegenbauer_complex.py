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
@pytest.mark.xfail(run=False)
def test_gegenbauer_complex(self):
    assert_mpmath_equal(lambda n, a, x: sc.eval_gegenbauer(int(n), a.real, x), exception_to_nan(mpmath.gegenbauer), [IntArg(0, 100), Arg(), ComplexArg()])