import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.stats import (binned_statistic, binned_statistic_2d,
from scipy._lib._util import check_random_state
from .common_tests import check_named_results
def test_gh14332(self):
    x = []
    size = 20
    for i in range(size):
        x += [1 - 0.1 ** i]
    bins = np.linspace(0, 1, 11)
    sum1, edges1, bc = binned_statistic_dd(x, np.ones(len(x)), bins=[bins], statistic='sum')
    sum2, edges2 = np.histogram(x, bins=bins)
    assert_allclose(sum1, sum2)
    assert_allclose(edges1[0], edges2)