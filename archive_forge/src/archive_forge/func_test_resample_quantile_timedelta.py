from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_resample_quantile_timedelta(unit):
    dtype = np.dtype(f'm8[{unit}]')
    df = DataFrame({'value': pd.to_timedelta(np.arange(4), unit='s').astype(dtype)}, index=pd.date_range('20200101', periods=4, tz='UTC'))
    result = df.resample('2D').quantile(0.99)
    expected = DataFrame({'value': [pd.Timedelta('0 days 00:00:00.990000'), pd.Timedelta('0 days 00:00:02.990000')]}, index=pd.date_range('20200101', periods=2, tz='UTC', freq='2D')).astype(dtype)
    tm.assert_frame_equal(result, expected)