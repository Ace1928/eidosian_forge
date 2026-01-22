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
def test_find_repeats(self):
    x = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]).astype('float')
    tmp = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]).astype('float')
    mask = tmp == 5.0
    xm = np.ma.array(tmp, mask=mask)
    x_orig, xm_orig = (x.copy(), xm.copy())
    r = stats.find_repeats(x)
    rm = stats.mstats.find_repeats(xm)
    assert_equal(r, rm)
    assert_equal(x, x_orig)
    assert_equal(xm, xm_orig)
    _, counts = stats.mstats.find_repeats([])
    assert_equal(counts, np.array(0, dtype=np.intp))