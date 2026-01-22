import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
@pytest.mark.xfail(run=False)
def test_tklmbda_pos_shape(self):
    _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(0, 100, inclusive_a=False)], spfunc_first=False, rtol=1e-05)