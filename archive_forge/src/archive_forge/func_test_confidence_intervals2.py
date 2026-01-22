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
@pytest.mark.parametrize('alternative, pval, ci_low, ci_high', [('less', 0.005656361, 0.0, 0.1872093), ('greater', 0.9987146, 0.008860761, 1.0), ('two-sided', 0.01191714, 0.006872485, 0.202706269)])
def test_confidence_intervals2(self, alternative, pval, ci_low, ci_high):
    res = stats.binomtest(3, n=50, p=0.2, alternative=alternative)
    assert_allclose(res.pvalue, pval, rtol=1e-06)
    assert_equal(res.statistic, 0.06)
    ci = res.proportion_ci(confidence_level=0.99)
    assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-06)