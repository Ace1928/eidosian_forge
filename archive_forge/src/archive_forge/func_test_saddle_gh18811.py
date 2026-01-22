import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.xfail
@pytest.mark.parametrize('case', ((lambda x: (x - 1) ** 3, 1), (lambda x: np.where(x > 1, (x - 1) ** 5, (x - 1) ** 3), 1)))
def test_saddle_gh18811(self, case):
    atol = 1e-16
    res = zeros._differentiate(*case, step_direction=[-1, 0, 1], atol=atol)
    assert np.all(res.success)
    assert_allclose(res.df, 0, atol=atol)