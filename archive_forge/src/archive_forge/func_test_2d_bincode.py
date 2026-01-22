import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_2d_bincode(self):
    x = self.x[:20]
    y = self.y[:20]
    v = self.v[:20]
    count1, binx1, biny1, bc = binned_statistic_2d(x, y, v, 'count', bins=3)
    bc2 = np.array([17, 11, 6, 16, 11, 17, 18, 17, 17, 7, 6, 18, 16, 6, 11, 16, 6, 6, 11, 8])
    bcount = [(bc == i).sum() for i in np.unique(bc)]
    assert_allclose(bc, bc2)
    count1adj = count1[count1.nonzero()]
    assert_allclose(bcount, count1adj)