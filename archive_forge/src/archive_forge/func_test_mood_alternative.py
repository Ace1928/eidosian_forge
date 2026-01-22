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
def test_mood_alternative(self):
    np.random.seed(0)
    x = stats.norm.rvs(scale=0.75, size=100)
    y = stats.norm.rvs(scale=1.25, size=100)
    stat1, p1 = stats.mood(x, y, alternative='two-sided')
    stat2, p2 = stats.mood(x, y, alternative='less')
    stat3, p3 = stats.mood(x, y, alternative='greater')
    assert stat1 == stat2 == stat3
    assert_allclose(p1, 0, atol=1e-07)
    assert_allclose(p2, p1 / 2)
    assert_allclose(p3, 1 - p1 / 2)
    with pytest.raises(ValueError, match='alternative must be...'):
        stats.mood(x, y, alternative='ekki-ekki')