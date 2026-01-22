import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
@pytest.mark.parametrize('test_func, numpy_func', [(stats.circmean, np.mean), (stats.circvar, np.var), (stats.circstd, np.std)])
def test_circfuncs_close(self, test_func, numpy_func):
    x = np.array([0.12675364631578953] * 10 + [0.12675365920187928] * 100)
    circstat = test_func(x)
    normal = numpy_func(x)
    assert_allclose(circstat, normal, atol=2e-08)