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
@pytest.mark.parametrize('x', [[355, 5, 2, 359, 10, 350, np.nan], np.array([[355, 5, 2, 359, 10, 350, np.nan], [351, 7, 4, 352, np.nan, 9, 349]])])
@pytest.mark.parametrize('test_func', [stats.circmean, stats.circvar, stats.circstd])
def test_nan_raise(self, test_func, x):
    assert_raises(ValueError, test_func, x, high=360, nan_policy='raise')