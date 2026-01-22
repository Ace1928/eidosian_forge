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
@pytest.mark.parametrize('test_func,expected', [(stats.circmean, {None: np.nan, 0: 355.66582264, 1: 0.28725053}), (stats.circvar, {None: np.nan, 0: 0.002570671054089924, 1: 0.005545914017677123}), (stats.circstd, {None: np.nan, 0: 4.11093193, 1: 6.04265394})])
def test_nan_propagate_array(self, test_func, expected):
    x = np.array([[355, 5, 2, 359, 10, 350, 1], [351, 7, 4, 352, 9, 349, np.nan], [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    for axis in expected.keys():
        out = test_func(x, high=360, axis=axis)
        if axis is None:
            assert_(np.isnan(out))
        else:
            assert_allclose(out[0], expected[axis], rtol=1e-07)
            assert_(np.isnan(out[1:]).all())