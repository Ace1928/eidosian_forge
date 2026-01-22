import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_empty_bins_std(self):
    x = self.x
    u = self.u
    print(binned_statistic(x, u, 'count', bins=1000))
    stat1, edges1, bc = binned_statistic(x, u, 'std', bins=1000)
    stat2, edges2, bc = binned_statistic(x, u, np.std, bins=1000)
    assert_allclose(stat1, stat2)