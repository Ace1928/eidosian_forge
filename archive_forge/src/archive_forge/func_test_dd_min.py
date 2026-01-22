import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_min(self):
    X = self.X
    v = self.v
    stat1, edges1, bc = binned_statistic_dd(X, v, 'min', bins=3)
    stat2, edges2, bc = binned_statistic_dd(X, v, np.min, bins=3)
    assert_allclose(stat1, stat2)
    assert_allclose(edges1, edges2)