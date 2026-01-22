from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_bws_input_validation(self):
    rng = np.random.default_rng(4571775098104213308)
    x, y = rng.random(size=(2, 7))
    message = '`x` and `y` must be exactly one-dimensional.'
    with pytest.raises(ValueError, match=message):
        stats.bws_test([x, x], [y, y])
    message = '`x` and `y` must not contain NaNs.'
    with pytest.raises(ValueError, match=message):
        stats.bws_test([np.nan], y)
    message = '`x` and `y` must be of nonzero size.'
    with pytest.raises(ValueError, match=message):
        stats.bws_test(x, [])
    message = 'alternative` must be one of...'
    with pytest.raises(ValueError, match=message):
        stats.bws_test(x, y, alternative='ekki-ekki')
    message = 'method` must be an instance of...'
    with pytest.raises(ValueError, match=message):
        stats.bws_test(x, y, method=42)