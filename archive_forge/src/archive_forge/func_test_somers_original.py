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
def test_somers_original(self):
    table = np.array([[8, 2], [6, 5], [3, 4], [1, 3], [2, 3]])
    table = table.T
    dyx = 129 / 340
    assert_allclose(stats.somersd(table).statistic, dyx)
    table = np.array([[25, 0], [85, 0], [0, 30]])
    dxy, dyx = (3300 / 5425, 3300 / 3300)
    assert_allclose(stats.somersd(table).statistic, dxy)
    assert_allclose(stats.somersd(table.T).statistic, dyx)
    table = np.array([[25, 0], [0, 30], [85, 0]])
    dyx = -1800 / 3300
    assert_allclose(stats.somersd(table.T).statistic, dyx)