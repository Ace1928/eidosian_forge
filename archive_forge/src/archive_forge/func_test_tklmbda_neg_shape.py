import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def test_tklmbda_neg_shape(self):
    _assert_inverts(sp.tklmbda, _tukey_lmbda_quantile, 0, [ProbArg(), Arg(-25, 0, inclusive_b=False)], spfunc_first=False, rtol=1e-05, endpt_atol=[1e-09, 1e-05])