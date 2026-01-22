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
def test_ks_2samp(self):
    """Checks that mstats.ks_2samp and stats.ks_2samp agree on masked arrays.
        gh-8431"""
    for mode in ['auto', 'exact', 'asymp']:
        with suppress_warnings() as sup:
            if mode in ['auto', 'exact']:
                message = 'ks_2samp: Exact calculation unsuccessful.'
                sup.filter(RuntimeWarning, message)
            for alternative in ['less', 'greater', 'two-sided']:
                for n in self.get_n():
                    x, y, xm, ym = self.generate_xy_sample(n)
                    res1 = stats.ks_2samp(x, y, alternative=alternative, mode=mode)
                    res2 = stats.mstats.ks_2samp(xm, ym, alternative=alternative, mode=mode)
                    assert_equal(np.asarray(res1), np.asarray(res2))
                    res3 = stats.ks_2samp(xm, y, alternative=alternative, mode=mode)
                    assert_equal(np.asarray(res1), np.asarray(res3))