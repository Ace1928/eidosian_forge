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
def test_small(self):
    x = [1, 2, 3, 3, 4]
    y = [3, 2, 6, 1, 6, 1, 4, 1]
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'Ties preclude use of exact statistic.')
        W, pval = stats.ansari(x, y)
    assert_almost_equal(W, 23.5, 11)
    assert_almost_equal(pval, 0.13499256881897437, 11)