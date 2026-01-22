import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_1d_multi_values(self):
    x = self.x
    v = self.v
    w = self.w
    stat1v, edges1v, bc1v = binned_statistic(x, v, 'mean', bins=10)
    stat1w, edges1w, bc1w = binned_statistic(x, w, 'mean', bins=10)
    stat2, edges2, bc2 = binned_statistic(x, [v, w], 'mean', bins=10)
    assert_allclose(stat2[0], stat1v)
    assert_allclose(stat2[1], stat1w)
    assert_allclose(edges1v, edges2)
    assert_allclose(bc1v, bc2)