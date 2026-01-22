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
def test_exact_U_equals_mean(self):
    res_l = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='less', method='exact')
    res_g = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='greater', method='exact')
    assert_equal(res_l.pvalue, res_g.pvalue)
    assert res_l.pvalue > 0.5
    res = mannwhitneyu([1, 2, 3], [1.5, 2.5], alternative='two-sided', method='exact')
    assert_equal(res, (3, 1))