from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_dst_anchor2(unit):
    dti = date_range('2013-09-30', '2013-11-02', freq='30Min', tz='Europe/Paris').as_unit(unit)
    values = range(dti.size)
    df = DataFrame({'a': values, 'b': values, 'c': values}, index=dti, dtype='int64')
    how = {'a': 'min', 'b': 'max', 'c': 'count'}
    rs = df.resample('W-MON')
    result = rs.agg(how)[['a', 'b', 'c']]
    expected = DataFrame({'a': [0, 48, 384, 720, 1056, 1394], 'b': [47, 383, 719, 1055, 1393, 1586], 'c': [48, 336, 336, 336, 338, 193]}, index=date_range('9/30/2013', '11/4/2013', freq='W-MON', tz='Europe/Paris').as_unit(unit))
    tm.assert_frame_equal(result, expected, 'W-MON Frequency')
    rs2 = df.resample('2W-MON')
    result2 = rs2.agg(how)[['a', 'b', 'c']]
    expected2 = DataFrame({'a': [0, 48, 720, 1394], 'b': [47, 719, 1393, 1586], 'c': [48, 672, 674, 193]}, index=date_range('9/30/2013', '11/11/2013', freq='2W-MON', tz='Europe/Paris').as_unit(unit))
    tm.assert_frame_equal(result2, expected2, '2W-MON Frequency')
    rs3 = df.resample('MS')
    result3 = rs3.agg(how)[['a', 'b', 'c']]
    expected3 = DataFrame({'a': [0, 48, 1538], 'b': [47, 1537, 1586], 'c': [48, 1490, 49]}, index=date_range('9/1/2013', '11/1/2013', freq='MS', tz='Europe/Paris').as_unit(unit))
    tm.assert_frame_equal(result3, expected3, 'MS Frequency')
    rs4 = df.resample('2MS')
    result4 = rs4.agg(how)[['a', 'b', 'c']]
    expected4 = DataFrame({'a': [0, 1538], 'b': [1537, 1586], 'c': [1538, 49]}, index=date_range('9/1/2013', '11/1/2013', freq='2MS', tz='Europe/Paris').as_unit(unit))
    tm.assert_frame_equal(result4, expected4, '2MS Frequency')
    df_daily = df['10/26/2013':'10/29/2013']
    rs_d = df_daily.resample('D')
    result_d = rs_d.agg({'a': 'min', 'b': 'max', 'c': 'count'})[['a', 'b', 'c']]
    expected_d = DataFrame({'a': [1248, 1296, 1346, 1394], 'b': [1295, 1345, 1393, 1441], 'c': [48, 50, 48, 48]}, index=date_range('10/26/2013', '10/29/2013', freq='D', tz='Europe/Paris').as_unit(unit))
    tm.assert_frame_equal(result_d, expected_d, 'D Frequency')