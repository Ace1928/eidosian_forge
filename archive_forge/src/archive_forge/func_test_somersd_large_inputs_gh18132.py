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
def test_somersd_large_inputs_gh18132(self):
    classes = [1, 2]
    n_samples = 10 ** 6
    random.seed(6272161)
    x = random.choices(classes, k=n_samples)
    y = random.choices(classes, k=n_samples)
    val_sklearn = -0.001528138777036947
    val_scipy = stats.somersd(x, y).statistic
    assert_allclose(val_sklearn, val_scipy, atol=1e-15)