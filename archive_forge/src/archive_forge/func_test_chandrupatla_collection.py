import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', [zeros.bisect, zeros.ridder, zeros.toms748])
def test_chandrupatla_collection(self, method):
    known_fail = {'fun7.4'} if method == zeros.ridder else {}
    self.run_collection('chandrupatla', method, method.__name__, known_fail=known_fail)