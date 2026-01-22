import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_1d_sum(self):
    x = self.x
    v = self.v
    sum1, edges1, bc = binned_statistic(x, v, 'sum', bins=10)
    sum2, edges2 = np.histogram(x, bins=10, weights=v)
    assert_allclose(sum1, sum2)
    assert_allclose(edges1, edges2)