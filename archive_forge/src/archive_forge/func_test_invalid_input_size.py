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
def test_invalid_input_size(self):
    assert_raises(ValueError, mstats.ttest_rel, np.arange(10), np.arange(11))
    x = np.arange(24)
    assert_raises(ValueError, mstats.ttest_rel, x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=1)
    assert_raises(ValueError, mstats.ttest_rel, x.reshape(2, 3, 4), x.reshape(2, 4, 3), axis=2)