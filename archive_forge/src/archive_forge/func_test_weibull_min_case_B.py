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
def test_weibull_min_case_B(self):
    x = np.array([74, 57, 48, 29, 502, 12, 70, 21, 29, 386, 59, 27, 153, 26, 326])
    message = 'Maximum likelihood estimation has converged to '
    with pytest.raises(ValueError, match=message):
        stats.anderson(x, 'weibull_min')