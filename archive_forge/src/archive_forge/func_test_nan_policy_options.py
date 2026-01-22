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
def test_nan_policy_options(self):
    x = [1, 2, np.nan]
    y = [4, 5, 6]
    mt1 = stats.median_test(x, y, nan_policy='propagate')
    s, p, m, t = stats.median_test(x, y, nan_policy='omit')
    assert_equal(mt1, (np.nan, np.nan, np.nan, None))
    assert_allclose(s, 0.31250000000000006)
    assert_allclose(p, 0.5761501220305787)
    assert_equal(m, 4.0)
    assert_equal(t, np.array([[0, 2], [2, 1]]))
    assert_raises(ValueError, stats.median_test, x, y, nan_policy='raise')