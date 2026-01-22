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
@pytest.mark.xfail(run=False, reason="it's not the same as the mpmath function --- maybe different definition?")
def test_pcfv(self):

    def pcfv(v, x):
        return sc.pbvv(v, x)[0]
    assert_mpmath_equal(pcfv, lambda v, x: time_limited()(exception_to_nan(mpmath.pcfv))(v, x, **HYPERKW), [Arg(), Arg()], n=1000)