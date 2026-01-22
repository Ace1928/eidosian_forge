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
@pytest.mark.parametrize('alternative, expected', [('two-sided', (1.01993853354993, 0.307757612977876)), ('less', (1.01993853354993, 1 - 0.153878806488938)), ('greater', (1.01993853354993, 0.153878806488938))])
def test_against_SAS_2(self, alternative, expected):
    x = [111, 107, 100, 99, 102, 106, 109, 108, 104, 99, 101, 96, 97, 102, 107, 113, 116, 113, 110, 98]
    y = [107, 108, 106, 98, 105, 103, 110, 105, 104, 100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99]
    res = stats.mood(x, y, alternative=alternative)
    assert_allclose(res, expected)