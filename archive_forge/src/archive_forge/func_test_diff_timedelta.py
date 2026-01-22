import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_timedelta(self, unit):
    df = DataFrame({'time': [Timestamp('20130101 9:01'), Timestamp('20130101 9:02')], 'value': [1.0, 2.0]})
    df['time'] = df['time'].dt.as_unit(unit)
    res = df.diff()
    exp = DataFrame([[pd.NaT, np.nan], [pd.Timedelta('00:01:00'), 1]], columns=['time', 'value'])
    exp['time'] = exp['time'].dt.as_unit(unit)
    tm.assert_frame_equal(res, exp)