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
def test_hurwitz_zeta(self):
    assert_mpmath_equal(sc.zeta, exception_to_nan(mpmath.zeta), [Arg(a=1, b=10000000000.0, inclusive_a=False), Arg(a=0, inclusive_a=False)])