import warnings
import platform
import numpy as np
from numpy import nan
import numpy.ma as ma
from numpy.ma import masked, nomask
import scipy.stats.mstats as mstats
from scipy import stats
from .common_tests import check_named_results
import pytest
from pytest import raises as assert_raises
from numpy.ma.testutils import (assert_equal, assert_almost_equal,
from numpy.testing import suppress_warnings
from scipy.stats import _mstats_basic
def test_sen_seasonal_slopes():
    rng = np.random.default_rng(5765986256978575148)
    x = rng.random(size=(100, 4))
    intra_slope, inter_slope = mstats.sen_seasonal_slopes(x)

    def dijk(yi):
        n = len(yi)
        x = np.arange(n)
        dy = yi - yi[:, np.newaxis]
        dx = x - x[:, np.newaxis]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return dy[mask] / dx[mask]
    for i in range(4):
        assert_allclose(np.median(dijk(x[:, i])), intra_slope[i])
    all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])
    assert_allclose(np.median(all_slopes), inter_slope)