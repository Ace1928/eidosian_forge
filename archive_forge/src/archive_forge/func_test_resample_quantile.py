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
def test_resample_quantile(series):
    ser = series
    q = 0.75
    freq = 'h'
    msg = 'Resampling with a PeriodIndex'
    warn = None
    if isinstance(series.index, PeriodIndex):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = ser.resample(freq).quantile(q)
        expected = ser.resample(freq).agg(lambda x: x.quantile(q)).rename(ser.name)
    tm.assert_series_equal(result, expected)