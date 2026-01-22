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
def test_large_sample(self):
    np.random.seed(4367)
    x = distributions.norm.rvs(size=1000000)
    y = distributions.norm.rvs(size=900000)
    r = cramervonmises_2samp(x, y)
    assert_(0 < r.pvalue < 1)
    r = cramervonmises_2samp(x, y + 0.1)
    assert_(0 < r.pvalue < 1)