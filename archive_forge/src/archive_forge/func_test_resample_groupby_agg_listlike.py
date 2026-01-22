from textwrap import dedent
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_resample_groupby_agg_listlike():
    ts = Timestamp('2021-02-28 00:00:00')
    df = DataFrame({'class': ['beta'], 'value': [69]}, index=Index([ts], name='date'))
    resampled = df.groupby('class').resample('ME')['value']
    result = resampled.agg(['sum', 'size'])
    expected = DataFrame([[69, 1]], index=pd.MultiIndex.from_tuples([('beta', ts)], names=['class', 'date']), columns=['sum', 'size'])
    tm.assert_frame_equal(result, expected)