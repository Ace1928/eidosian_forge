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
def test_chi(self):

    def chi(x):
        return sc.shichi(x)[1]
    assert_mpmath_equal(chi, mpmath.chi, [Arg()])
    assert_mpmath_equal(chi, mpmath.chi, [FixedArg([88 - 1e-09, 88, 88 + 1e-09])])