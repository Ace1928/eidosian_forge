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
@all_ts
@pytest.mark.parametrize('freq', [pytest.param('ME', marks=pytest.mark.xfail(reason="Don't know why this fails")), 'D', 'h'])
def test_resample_nat_index_series(freq, series, resample_method):
    ser = series.copy()
    ser.index = PeriodIndex([NaT] * len(ser), freq=freq)
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs = ser.resample(freq)
    result = getattr(rs, resample_method)()
    if resample_method == 'ohlc':
        expected = DataFrame([], index=ser.index[:0].copy(), columns=['open', 'high', 'low', 'close'])
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser[:0].copy()
        tm.assert_series_equal(result, expected, check_dtype=False)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq