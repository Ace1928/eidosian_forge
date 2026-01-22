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
@pytest.mark.parametrize('alternative, pval, ci_low', [('less', 1.0, 0.0), ('greater', 9.536743e-07, 0.7411344), ('two-sided', 9.536743e-07, 0.6915029)])
def test_confidence_interval_exact_k_is_n(self, alternative, pval, ci_low):
    res = stats.binomtest(10, 10, p=0.25, alternative=alternative)
    assert_allclose(res.pvalue, pval, rtol=1e-06)
    ci = res.proportion_ci(confidence_level=0.95)
    assert_equal(ci.high, 1.0)
    assert_allclose(ci.low, ci_low, rtol=1e-06)