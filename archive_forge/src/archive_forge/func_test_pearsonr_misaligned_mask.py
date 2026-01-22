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
def test_pearsonr_misaligned_mask(self):
    mx = np.ma.masked_array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 0, 0, 0])
    my = np.ma.masked_array([9, 8, 7, 6, 5, 9], mask=[0, 0, 1, 0, 0, 0])
    x = np.array([1, 4, 5, 6])
    y = np.array([9, 6, 5, 9])
    mr, mp = mstats.pearsonr(mx, my)
    r, p = stats.pearsonr(x, y)
    assert_equal(mr, r)
    assert_equal(mp, p)