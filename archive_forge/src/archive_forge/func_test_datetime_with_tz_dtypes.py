from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_datetime_with_tz_dtypes(self):
    tzframe = DataFrame({'A': date_range('20130101', periods=3), 'B': date_range('20130101', periods=3, tz='US/Eastern'), 'C': date_range('20130101', periods=3, tz='CET')})
    tzframe.iloc[1, 1] = pd.NaT
    tzframe.iloc[1, 2] = pd.NaT
    result = tzframe.dtypes.sort_index()
    expected = Series([np.dtype('datetime64[ns]'), DatetimeTZDtype('ns', 'US/Eastern'), DatetimeTZDtype('ns', 'CET')], ['A', 'B', 'C'])
    tm.assert_series_equal(result, expected)