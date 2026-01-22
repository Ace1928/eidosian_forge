import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_zero_dedges(self):
    x = np.random.random((10000, 3))
    v = np.random.random(10000)
    bins = np.linspace(0, 1, 10)
    bins = np.append(bins, 1)
    bins = (bins, bins, bins)
    with assert_raises(ValueError, match='difference is numerically 0'):
        binned_statistic_dd(x, v, 'mean', bins=bins)