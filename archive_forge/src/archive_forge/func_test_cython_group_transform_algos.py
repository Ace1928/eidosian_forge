import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_cython_group_transform_algos():
    is_datetimelike = False
    labels = np.array([0, 0, 0, 0, 0], dtype=np.intp)
    ngroups = 1
    data = np.array([[1], [2], [3], [np.nan], [4]], dtype='float64')
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumprod(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 2, 6, np.nan, 24], dtype='float64')
    tm.assert_numpy_array_equal(actual[:, 0], expected)
    actual = np.zeros_like(data)
    actual.fill(np.nan)
    group_cumsum(actual, data, labels, ngroups, is_datetimelike)
    expected = np.array([1, 3, 6, np.nan, 10], dtype='float64')
    tm.assert_numpy_array_equal(actual[:, 0], expected)
    is_datetimelike = True
    data = np.array([np.timedelta64(1, 'ns')] * 5, dtype='m8[ns]')[:, None]
    actual = np.zeros_like(data, dtype='int64')
    group_cumsum(actual, data.view('int64'), labels, ngroups, is_datetimelike)
    expected = np.array([np.timedelta64(1, 'ns'), np.timedelta64(2, 'ns'), np.timedelta64(3, 'ns'), np.timedelta64(4, 'ns'), np.timedelta64(5, 'ns')])
    tm.assert_numpy_array_equal(actual[:, 0].view('m8[ns]'), expected)