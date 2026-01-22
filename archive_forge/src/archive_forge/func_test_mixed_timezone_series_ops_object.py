import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_mixed_timezone_series_ops_object(self):
    ser = Series([Timestamp('2015-01-01', tz='US/Eastern'), Timestamp('2015-01-01', tz='Asia/Tokyo')], name='xxx')
    assert ser.dtype == object
    exp = Series([Timestamp('2015-01-02', tz='US/Eastern'), Timestamp('2015-01-02', tz='Asia/Tokyo')], name='xxx')
    tm.assert_series_equal(ser + pd.Timedelta('1 days'), exp)
    tm.assert_series_equal(pd.Timedelta('1 days') + ser, exp)
    ser2 = Series([Timestamp('2015-01-03', tz='US/Eastern'), Timestamp('2015-01-05', tz='Asia/Tokyo')], name='xxx')
    assert ser2.dtype == object
    exp = Series([pd.Timedelta('2 days'), pd.Timedelta('4 days')], name='xxx', dtype=object)
    tm.assert_series_equal(ser2 - ser, exp)
    tm.assert_series_equal(ser - ser2, -exp)
    ser = Series([pd.Timedelta('01:00:00'), pd.Timedelta('02:00:00')], name='xxx', dtype=object)
    assert ser.dtype == object
    exp = Series([pd.Timedelta('01:30:00'), pd.Timedelta('02:30:00')], name='xxx', dtype=object)
    tm.assert_series_equal(ser + pd.Timedelta('00:30:00'), exp)
    tm.assert_series_equal(pd.Timedelta('00:30:00') + ser, exp)