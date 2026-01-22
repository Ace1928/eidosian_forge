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
def test_airyai(self):
    assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-100000000.0, 100000000.0)], rtol=1e-05)
    assert_mpmath_equal(lambda z: sc.airy(z)[0], mpmath.airyai, [Arg(-1000.0, 1000.0)])