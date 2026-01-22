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
@pytest.mark.parametrize('input_sample,expected', [([[0, 5], [0, 10]], (np.nan, np.nan)), ([[5, 0], [10, 0]], (np.nan, np.nan))])
def test_row_or_col_zero(self, input_sample, expected):
    res = boschloo_exact(input_sample)
    statistic, pvalue = (res.statistic, res.pvalue)
    assert_equal(pvalue, expected[0])
    assert_equal(statistic, expected[1])