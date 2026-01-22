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
def test_equal_mean_median(self):
    x = np.linspace(-1, 1, 21)
    np.random.seed(1234)
    x2 = np.random.permutation(x)
    y = x ** 3
    W1, pval1 = stats.levene(x, y, center='mean')
    W2, pval2 = stats.levene(x2, y, center='median')
    assert_almost_equal(W1, W2)
    assert_almost_equal(pval1, pval2)