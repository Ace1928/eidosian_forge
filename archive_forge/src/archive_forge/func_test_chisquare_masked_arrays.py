import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_chisquare_masked_arrays():
    obs = np.array([[8, 8, 16, 32, -1], [-1, -1, 3, 4, 5]]).T
    mask = np.array([[0, 0, 0, 0, 1], [1, 1, 0, 0, 0]]).T
    mobs = np.ma.masked_array(obs, mask)
    expected_chisq = np.array([24.0, 0.5])
    expected_g = np.array([2 * (2 * 8 * np.log(0.5) + 32 * np.log(2.0)), 2 * (3 * np.log(0.75) + 5 * np.log(1.25))])
    chi2 = stats.distributions.chi2
    chisq, p = stats.chisquare(mobs)
    mat.assert_array_equal(chisq, expected_chisq)
    mat.assert_array_almost_equal(p, chi2.sf(expected_chisq, mobs.count(axis=0) - 1))
    g, p = stats.power_divergence(mobs, lambda_='log-likelihood')
    mat.assert_array_almost_equal(g, expected_g, decimal=15)
    mat.assert_array_almost_equal(p, chi2.sf(expected_g, mobs.count(axis=0) - 1))
    chisq, p = stats.chisquare(mobs.T, axis=1)
    mat.assert_array_equal(chisq, expected_chisq)
    mat.assert_array_almost_equal(p, chi2.sf(expected_chisq, mobs.T.count(axis=1) - 1))
    g, p = stats.power_divergence(mobs.T, axis=1, lambda_='log-likelihood')
    mat.assert_array_almost_equal(g, expected_g, decimal=15)
    mat.assert_array_almost_equal(p, chi2.sf(expected_g, mobs.count(axis=0) - 1))
    obs1 = np.ma.array([3, 5, 6, 99, 10], mask=[0, 0, 0, 1, 0])
    exp1 = np.ma.array([2, 4, 8, 10, 99], mask=[0, 0, 0, 0, 1])
    chi2, p = stats.chisquare(obs1, f_exp=exp1)
    mat.assert_array_equal(chi2, 1 / 2 + 1 / 4 + 4 / 8)
    chisq, p = stats.chisquare(np.ma.array([1, 2, 3]), axis=None)
    assert_(isinstance(chisq, np.float64))
    assert_(isinstance(p, np.float64))
    assert_equal(chisq, 1.0)
    assert_almost_equal(p, stats.distributions.chi2.sf(1.0, 2))
    with np.errstate(invalid='ignore'):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'Mean of empty slice')
            chisq, p = stats.chisquare(np.ma.array([]))
    assert_(isinstance(chisq, np.ma.MaskedArray))
    assert_equal(chisq.shape, ())
    assert_(chisq.mask)
    empty3 = np.ma.array([[], [], []])
    chisq, p = stats.chisquare(empty3)
    assert_(isinstance(chisq, np.ma.MaskedArray))
    mat.assert_array_equal(chisq, [])
    with np.errstate(invalid='ignore'):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'Mean of empty slice')
            chisq, p = stats.chisquare(empty3.T)
    assert_(isinstance(chisq, np.ma.MaskedArray))
    assert_equal(chisq.shape, (3,))
    assert_(np.all(chisq.mask))