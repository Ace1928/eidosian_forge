from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat
@pytest.mark.parametrize('_index_factory,_series_name,_index_start,_index_end', [DATE_RANGE, TIMEDELTA_RANGE])
def test_asfreq_fill_value(series, create_index):
    ser = series
    result = ser.resample('1h').asfreq()
    new_index = create_index(ser.index[0], ser.index[-1], freq='1h')
    expected = ser.reindex(new_index)
    tm.assert_series_equal(result, expected)
    frame = ser.astype('float').to_frame('value')
    frame.iloc[1] = None
    result = frame.resample('1h').asfreq(fill_value=4.0)
    new_index = create_index(frame.index[0], frame.index[-1], freq='1h')
    expected = frame.reindex(new_index, fill_value=4.0)
    tm.assert_frame_equal(result, expected)