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
def test_resampler_is_iterable(series):
    freq = 'h'
    tg = Grouper(freq=freq, convention='start')
    msg = 'Resampling with a PeriodIndex'
    warn = None
    if isinstance(series.index, PeriodIndex):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        grouped = series.groupby(tg)
    with tm.assert_produces_warning(warn, match=msg):
        resampled = series.resample(freq)
    for (rk, rv), (gk, gv) in zip(resampled, grouped):
        assert rk == gk
        tm.assert_series_equal(rv, gv)