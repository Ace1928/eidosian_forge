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
def test_weibull_min_case_A(self):
    x = np.array([225, 171, 198, 189, 189, 135, 162, 135, 117, 162])
    res = stats.anderson(x, 'weibull_min')
    m, loc, scale = res.fit_result.params
    assert_allclose((m, loc, scale), (2.38, 99.02, 78.23), rtol=0.002)
    assert_allclose(res.statistic, 0.26, rtol=0.001)
    assert res.statistic < res.critical_values[0]
    c = 1 / m
    assert_allclose(c, 1 / 2.38, rtol=0.002)
    As40 = _Avals_weibull[-3]
    As45 = _Avals_weibull[-2]
    As_ref = As40 + (c - 0.4) / (0.45 - 0.4) * (As45 - As40)
    assert np.all(res.critical_values > As_ref)
    assert_allclose(res.critical_values, As_ref, atol=0.001)