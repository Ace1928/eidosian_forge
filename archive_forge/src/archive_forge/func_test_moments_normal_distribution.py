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
def test_moments_normal_distribution(self):
    np.random.seed(32149)
    data = np.random.randn(12345)
    moments = [stats.kstat(data, n) for n in [1, 2, 3, 4]]
    expected = [0.011315, 1.017931, 0.05811052, 0.0754134]
    assert_allclose(moments, expected, rtol=0.0001)
    m1 = stats.moment(data, moment=1)
    m2 = stats.moment(data, moment=2)
    m3 = stats.moment(data, moment=3)
    assert_allclose((m1, m2, m3), expected[:-1], atol=0.02, rtol=0.01)