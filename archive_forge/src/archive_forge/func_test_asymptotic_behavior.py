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
def test_asymptotic_behavior(self):
    np.random.seed(0)
    x = np.random.rand(5)
    y = np.random.rand(5)
    res1 = mannwhitneyu(x, y, method='exact')
    res2 = mannwhitneyu(x, y, method='asymptotic')
    assert res1.statistic == res2.statistic
    assert np.abs(res1.pvalue - res2.pvalue) > 0.01
    x = np.random.rand(40)
    y = np.random.rand(40)
    res1 = mannwhitneyu(x, y, method='exact')
    res2 = mannwhitneyu(x, y, method='asymptotic')
    assert res1.statistic == res2.statistic
    assert np.abs(res1.pvalue - res2.pvalue) < 0.001