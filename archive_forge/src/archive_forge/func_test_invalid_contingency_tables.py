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
def test_invalid_contingency_tables(self):
    N = 100
    shape = (4, 6)
    size = np.prod(shape)
    np.random.seed(0)
    s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
    s5 = s - 2
    message = 'All elements of the contingency table must be non-negative'
    with assert_raises(ValueError, match=message):
        stats.somersd(s5)
    s6 = s + 0.01
    message = 'All elements of the contingency table must be integer'
    with assert_raises(ValueError, match=message):
        stats.somersd(s6)
    message = 'At least two elements of the contingency table must be nonzero.'
    with assert_raises(ValueError, match=message):
        stats.somersd([[]])
    with assert_raises(ValueError, match=message):
        stats.somersd([[1]])
    s7 = np.zeros((3, 3))
    with assert_raises(ValueError, match=message):
        stats.somersd(s7)
    s7[0, 1] = 1
    with assert_raises(ValueError, match=message):
        stats.somersd(s7)