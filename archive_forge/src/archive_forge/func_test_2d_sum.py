import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_sum(self):
    x = self.x
    y = self.y
    v = self.v
    sum1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'sum', bins=5)
    sum2, binx2, biny2 = np.histogram2d(x, y, bins=5, weights=v)
    assert_allclose(sum1, sum2)
    assert_allclose(binx1, binx2)
    assert_allclose(biny1, biny2)