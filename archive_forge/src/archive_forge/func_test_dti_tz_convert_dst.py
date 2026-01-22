from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_dti_tz_convert_dst(self):
    for freq, n in [('h', 1), ('min', 60), ('s', 3600)]:
        idx = date_range('2014-03-08 23:00', '2014-03-09 09:00', freq=freq, tz='UTC')
        idx = idx.tz_convert('US/Eastern')
        expected = np.repeat(np.array([18, 19, 20, 21, 22, 23, 0, 1, 3, 4, 5]), np.array([n, n, n, n, n, n, n, n, n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
        idx = date_range('2014-03-08 18:00', '2014-03-09 05:00', freq=freq, tz='US/Eastern')
        idx = idx.tz_convert('UTC')
        expected = np.repeat(np.array([23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([n, n, n, n, n, n, n, n, n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
        idx = date_range('2014-11-01 23:00', '2014-11-02 09:00', freq=freq, tz='UTC')
        idx = idx.tz_convert('US/Eastern')
        expected = np.repeat(np.array([19, 20, 21, 22, 23, 0, 1, 1, 2, 3, 4]), np.array([n, n, n, n, n, n, n, n, n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
        idx = date_range('2014-11-01 18:00', '2014-11-02 05:00', freq=freq, tz='US/Eastern')
        idx = idx.tz_convert('UTC')
        expected = np.repeat(np.array([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([n, n, n, n, n, n, n, n, n, n, n, n, 1]))
        tm.assert_index_equal(idx.hour, Index(expected, dtype=np.int32))
    idx = date_range('2014-03-08 00:00', '2014-03-09 00:00', freq='D', tz='UTC')
    idx = idx.tz_convert('US/Eastern')
    tm.assert_index_equal(idx.hour, Index([19, 19], dtype=np.int32))
    idx = date_range('2014-03-08 00:00', '2014-03-09 00:00', freq='D', tz='US/Eastern')
    idx = idx.tz_convert('UTC')
    tm.assert_index_equal(idx.hour, Index([5, 5], dtype=np.int32))
    idx = date_range('2014-11-01 00:00', '2014-11-02 00:00', freq='D', tz='UTC')
    idx = idx.tz_convert('US/Eastern')
    tm.assert_index_equal(idx.hour, Index([20, 20], dtype=np.int32))
    idx = date_range('2014-11-01 00:00', '2014-11-02 000:00', freq='D', tz='US/Eastern')
    idx = idx.tz_convert('UTC')
    tm.assert_index_equal(idx.hour, Index([4, 4], dtype=np.int32))