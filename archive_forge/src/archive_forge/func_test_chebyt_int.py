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
def test_chebyt_int(self):
    assert_mpmath_equal(lambda n, x: sc.eval_chebyt(int(n), x), exception_to_nan(lambda n, x: mpmath.chebyt(n, x, **HYPERKW)), [IntArg(), Arg()], dps=50)