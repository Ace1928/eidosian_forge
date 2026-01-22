import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_dd_binnumbers_unraveled(self):
    X = self.X
    v = self.v
    stat, edgesx, bcx = binned_statistic(X[:, 0], v, 'mean', bins=15)
    stat, edgesy, bcy = binned_statistic(X[:, 1], v, 'mean', bins=20)
    stat, edgesz, bcz = binned_statistic(X[:, 2], v, 'mean', bins=10)
    stat2, edges2, bc2 = binned_statistic_dd(X, v, 'mean', bins=(15, 20, 10), expand_binnumbers=True)
    assert_allclose(bcx, bc2[0])
    assert_allclose(bcy, bc2[1])
    assert_allclose(bcz, bc2[2])