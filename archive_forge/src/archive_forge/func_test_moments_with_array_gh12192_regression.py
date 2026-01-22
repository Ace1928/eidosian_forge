import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def test_moments_with_array_gh12192_regression():
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=1)
    expected0 = np.array([1.0, 2.0, 3.0])
    npt.assert_equal(vals0, expected0)
    vals1 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=-1)
    expected1 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals1, expected1)
    vals2 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[-3, 1, 0])
    expected2 = np.array([np.nan, 2.0, np.nan])
    npt.assert_equal(vals2, expected2)
    vals3 = stats.norm.moment(order=2, loc=0, scale=-4)
    expected3 = np.nan
    npt.assert_equal(vals3, expected3)
    assert isinstance(vals3, expected3.__class__)
    vals4 = stats.norm.moment(order=2, loc=[1, 0, 2], scale=[3, -4, -5])
    expected4 = np.array([10.0, np.nan, np.nan])
    npt.assert_equal(vals4, expected4)
    vals5 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[5.0, -2, 100.0])
    expected5 = np.array([25.0, np.nan, 10000.0])
    npt.assert_equal(vals5, expected5)
    vals6 = stats.norm.moment(order=2, loc=[0, 0, 0], scale=[-5.0, -2, -100.0])
    expected6 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals6, expected6)
    vals7 = stats.chi.moment(order=2, df=1, loc=0, scale=0)
    expected7 = np.nan
    npt.assert_equal(vals7, expected7)
    assert isinstance(vals7, expected7.__class__)
    vals8 = stats.chi.moment(order=2, df=[1, 2, 3], loc=0, scale=0)
    expected8 = np.array([np.nan, np.nan, np.nan])
    npt.assert_equal(vals8, expected8)
    vals9 = stats.chi.moment(order=2, df=[1, 2, 3], loc=[1.0, 0.0, 2.0], scale=[1.0, -3.0, 0.0])
    expected9 = np.array([3.59576912, np.nan, np.nan])
    npt.assert_allclose(vals9, expected9, rtol=1e-08)
    vals10 = stats.norm.moment(5, [1.0, 2.0], [1.0, 2.0])
    expected10 = np.array([26.0, 832.0])
    npt.assert_allclose(vals10, expected10, rtol=1e-13)
    a = [-1.1, 0, 1, 2.2, np.pi]
    b = [-1.1, 0, 1, 2.2, np.pi]
    loc = [-1.1, 0, np.sqrt(2)]
    scale = [-2.1, 0, 1, 2.2, np.pi]
    a = np.array(a).reshape((-1, 1, 1, 1))
    b = np.array(b).reshape((-1, 1, 1))
    loc = np.array(loc).reshape((-1, 1))
    scale = np.array(scale)
    vals11 = stats.beta.moment(order=2, a=a, b=b, loc=loc, scale=scale)
    a, b, loc, scale = np.broadcast_arrays(a, b, loc, scale)
    for i in np.ndenumerate(a):
        with np.errstate(invalid='ignore', divide='ignore'):
            i = i[0]
            expected = stats.beta.moment(order=2, a=a[i], b=b[i], loc=loc[i], scale=scale[i])
            np.testing.assert_equal(vals11[i], expected)