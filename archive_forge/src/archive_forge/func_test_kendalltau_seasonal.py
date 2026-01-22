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
def test_kendalltau_seasonal(self):
    x = [[nan, nan, 4, 2, 16, 26, 5, 1, 5, 1, 2, 3, 1], [4, 3, 5, 3, 2, 7, 3, 1, 1, 2, 3, 5, 3], [3, 2, 5, 6, 18, 4, 9, 1, 1, nan, 1, 1, nan], [nan, 6, 11, 4, 17, nan, 6, 1, 1, 2, 5, 1, 1]]
    x = ma.fix_invalid(x).T
    output = mstats.kendalltau_seasonal(x)
    assert_almost_equal(output['global p-value (indep)'], 0.008, 3)
    assert_almost_equal(output['seasonal p-value'].round(2), [0.18, 0.53, 0.2, 0.04])