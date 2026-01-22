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
def test_epps_singleton_array_like(self):
    np.random.seed(1234)
    x, y = (np.arange(30), np.arange(28))
    w1, p1 = epps_singleton_2samp(list(x), list(y))
    w2, p2 = epps_singleton_2samp(tuple(x), tuple(y))
    w3, p3 = epps_singleton_2samp(x, y)
    assert_(w1 == w2 == w3)
    assert_(p1 == p2 == p3)