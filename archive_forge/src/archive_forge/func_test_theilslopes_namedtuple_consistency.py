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
def test_theilslopes_namedtuple_consistency():
    """
    Simple test to ensure tuple backwards-compatibility of the returned
    TheilslopesResult object
    """
    y = [1, 2, 4]
    x = [4, 6, 8]
    slope, intercept, low_slope, high_slope = mstats.theilslopes(y, x)
    result = mstats.theilslopes(y, x)
    assert_equal(slope, result.slope)
    assert_equal(intercept, result.intercept)
    assert_equal(low_slope, result.low_slope)
    assert_equal(high_slope, result.high_slope)