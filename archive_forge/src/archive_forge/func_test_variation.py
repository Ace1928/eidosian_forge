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
def test_variation(self):
    for n in self.get_n():
        x, y, xm, ym = self.generate_xy_sample(n)
        assert_almost_equal(stats.variation(x), stats.mstats.variation(xm), decimal=12)
        assert_almost_equal(stats.variation(y), stats.mstats.variation(ym), decimal=12)