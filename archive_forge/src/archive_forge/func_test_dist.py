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
def test_dist(self):
    x = stats.tukeylambda.rvs(-0.7, loc=2, scale=0.5, size=10000, random_state=1234567) + 10000.0
    max1 = stats.ppcc_max(x, dist='tukeylambda')
    max2 = stats.ppcc_max(x, dist=stats.tukeylambda)
    assert_almost_equal(max1, -0.7121536652126415, decimal=5)
    assert_almost_equal(max2, -0.7121536652126415, decimal=5)
    max3 = stats.ppcc_max(x)
    assert_almost_equal(max3, -0.7121536652126415, decimal=5)