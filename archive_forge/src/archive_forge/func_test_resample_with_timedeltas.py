from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
def test_resample_with_timedeltas():
    expected = DataFrame({'A': np.arange(1480)})
    expected = expected.groupby(expected.index // 30).sum()
    expected.index = timedelta_range('0 days', freq='30min', periods=50)
    df = DataFrame({'A': np.arange(1480)}, index=pd.to_timedelta(np.arange(1480), unit='min'))
    result = df.resample('30min').sum()
    tm.assert_frame_equal(result, expected)
    s = df['A']
    result = s.resample('30min').sum()
    tm.assert_series_equal(result, expected['A'])