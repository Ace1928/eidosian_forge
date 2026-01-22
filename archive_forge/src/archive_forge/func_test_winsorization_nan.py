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
def test_winsorization_nan(self):
    data = ma.array([np.nan, np.nan, 0, 1, 2])
    assert_raises(ValueError, mstats.winsorize, data, (0.05, 0.05), nan_policy='raise')
    assert_equal(mstats.winsorize(data, (0.4, 0.4)), ma.array([2, 2, 2, 2, 2]))
    assert_equal(mstats.winsorize(data, (0.8, 0.8)), ma.array([np.nan, np.nan, np.nan, np.nan, np.nan]))
    assert_equal(mstats.winsorize(data, (0.4, 0.4), nan_policy='omit'), ma.array([np.nan, np.nan, 2, 2, 2]))
    assert_equal(mstats.winsorize(data, (0.8, 0.8), nan_policy='omit'), ma.array([np.nan, np.nan, 2, 2, 2]))