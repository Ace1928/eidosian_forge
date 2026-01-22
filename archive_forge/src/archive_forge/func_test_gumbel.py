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
def test_gumbel(self):
    v = np.ones(100)
    v[0] = 0.0
    a2, crit, sig = stats.anderson(v, 'gumbel')
    n = len(v)
    xbar, s = stats.gumbel_l.fit(v)
    logcdf = stats.gumbel_l.logcdf(v, xbar, s)
    logsf = stats.gumbel_l.logsf(v, xbar, s)
    i = np.arange(1, n + 1)
    expected_a2 = -n - np.mean((2 * i - 1) * (logcdf + logsf[::-1]))
    assert_allclose(a2, expected_a2)