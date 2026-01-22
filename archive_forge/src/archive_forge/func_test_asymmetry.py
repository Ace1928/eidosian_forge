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
def test_asymmetry(self):
    x = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    d_cr = 0.27272727272727
    d_rc = 0.34285714285714
    p = 0.0928919408837
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, d_cr, atol=1e-15)
    assert_allclose(res.pvalue, p, atol=0.0001)
    assert_equal(res.table.shape, (3, 2))
    res = stats.somersd(y, x)
    assert_allclose(res.statistic, d_rc, atol=1e-15)
    assert_allclose(res.pvalue, p, atol=1e-15)
    assert_equal(res.table.shape, (2, 3))