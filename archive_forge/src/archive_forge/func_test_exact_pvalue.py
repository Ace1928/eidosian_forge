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
@pytest.mark.parametrize('statistic, m, n, pval', [(710, 5, 6, 48.0 / 462), (1897, 7, 7, 117.0 / 1716), (576, 4, 6, 2.0 / 210), (1764, 6, 7, 2.0 / 1716)])
def test_exact_pvalue(self, statistic, m, n, pval):
    assert_equal(_pval_cvm_2samp_exact(statistic, m, n), pval)