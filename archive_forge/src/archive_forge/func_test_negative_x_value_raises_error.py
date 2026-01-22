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
@pytest.mark.parametrize('bad_x', [np.array([1, -42, 12345.6]), np.array([np.nan, 42, 1])])
def test_negative_x_value_raises_error(self, bad_x):
    """Test boxcox_normmax raises ValueError if x contains non-positive values."""
    message = 'only positive, finite, real numbers'
    with pytest.raises(ValueError, match=message):
        stats.boxcox_normmax(bad_x)