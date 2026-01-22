import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_1d_range_keyword(self):
    np.random.seed(9865)
    x = np.arange(30)
    data = np.random.random(30)
    mean, bins, _ = binned_statistic(x[:15], data[:15])
    mean_range, bins_range, _ = binned_statistic(x, data, range=[(0, 14)])
    mean_range2, bins_range2, _ = binned_statistic(x, data, range=(0, 14))
    assert_allclose(mean, mean_range)
    assert_allclose(bins, bins_range)
    assert_allclose(mean, mean_range2)
    assert_allclose(bins, bins_range2)