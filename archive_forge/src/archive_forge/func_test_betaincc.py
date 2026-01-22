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
def test_betaincc(self):
    assert_mpmath_equal(sc.betaincc, time_limited()(exception_to_nan(lambda a, b, x: mpmath.betainc(a, b, x, 1, regularized=True))), [Arg(), Arg(), Arg()], dps=400)