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
def test_trimr(self):
    x = ma.arange(10)
    result = mstats.trimr(x, limits=(0.15, 0.14), inclusive=(False, False))
    expected = ma.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mask=[1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    assert_equal(result, expected)
    assert_equal(result.mask, expected.mask)