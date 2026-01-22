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
@pytest.mark.parametrize('freq', ['ME', 'D', 'h'])
def test_resample_empty_series(freq, empty_series_dti, resample_method):
    ser = empty_series_dti
    if freq == 'ME' and isinstance(ser.index, TimedeltaIndex):
        msg = "Resampling on a TimedeltaIndex requires fixed-duration `freq`, e.g. '24h' or '3D', not <MonthEnd>"
        with pytest.raises(ValueError, match=msg):
            ser.resample(freq)
        return
    elif freq == 'ME' and isinstance(ser.index, PeriodIndex):
        freq = 'M'
    warn = None
    if isinstance(ser.index, PeriodIndex):
        warn = FutureWarning
    msg = 'Resampling with a PeriodIndex is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        rs = ser.resample(freq)
    result = getattr(rs, resample_method)()
    if resample_method == 'ohlc':
        expected = DataFrame([], index=ser.index[:0].copy(), columns=['open', 'high', 'low', 'close'])
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_frame_equal(result, expected, check_dtype=False)
    else:
        expected = ser.copy()
        expected.index = _asfreq_compat(ser.index, freq)
        tm.assert_series_equal(result, expected, check_dtype=False)
    tm.assert_index_equal(result.index, expected.index)
    assert result.index.freq == expected.index.freq