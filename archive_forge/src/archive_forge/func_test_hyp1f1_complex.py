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
def test_hyp1f1_complex(self):
    assert_mpmath_equal(inf_to_nan(lambda a, b, x: sc.hyp1f1(a.real, b.real, x)), exception_to_nan(lambda a, b, x: mpmath.hyp1f1(a, b, x, **HYPERKW)), [Arg(-1000.0, 1000.0), Arg(-1000.0, 1000.0), ComplexArg()], n=2000)