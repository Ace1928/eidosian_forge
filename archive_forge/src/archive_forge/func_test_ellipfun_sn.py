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
def test_ellipfun_sn(self):

    def sn(u, m):
        if u == 0:
            return 0
        else:
            return mpmath.ellipfun('sn', u=u, m=m)
    assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[0], sn, [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)