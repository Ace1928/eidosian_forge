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
def test_basic_with_axis(self):
    a = np.ma.masked_array([[0, 1, 2, 3, 4, 9], [5, 5, 0, 9, 3, 3]], mask=[[0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]])
    result = mstats.describe(a, axis=1)
    assert_equal(result.nobs, [5, 4])
    amin, amax = result.minmax
    assert_equal(amin, [0, 3])
    assert_equal(amax, [4, 5])
    assert_equal(result.mean, [2.0, 4.0])
    assert_equal(result.variance, [2.0, 1.0])
    assert_equal(result.skewness, [0.0, 0.0])
    assert_allclose(result.kurtosis, [-1.3, -2.0])