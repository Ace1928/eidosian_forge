import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def test_tklmbda_zero_shape(self):
    one = mpmath.mpf(1)
    assert_mpmath_equal(lambda x: sp.tklmbda(x, 0), lambda x: one / (mpmath.exp(-x) + one), [Arg()], rtol=1e-07)