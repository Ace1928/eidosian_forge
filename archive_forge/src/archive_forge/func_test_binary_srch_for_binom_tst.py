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
def test_binary_srch_for_binom_tst(self):
    n = 10
    p = 0.5
    k = 3
    i = np.arange(np.ceil(p * n), n + 1)
    d = stats.binom.pmf(k, n, p)
    y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
    ix = _binary_search_for_binom_tst(lambda x1: -stats.binom.pmf(x1, n, p), -d, np.ceil(p * n), n)
    y2 = n - ix + int(d == stats.binom.pmf(ix, n, p))
    assert_allclose(y1, y2, rtol=1e-09)
    k = 7
    i = np.arange(np.floor(p * n) + 1)
    d = stats.binom.pmf(k, n, p)
    y1 = np.sum(stats.binom.pmf(i, n, p) <= d, axis=0)
    ix = _binary_search_for_binom_tst(lambda x1: stats.binom.pmf(x1, n, p), d, 0, np.floor(p * n))
    y2 = ix + 1
    assert_allclose(y1, y2, rtol=1e-09)