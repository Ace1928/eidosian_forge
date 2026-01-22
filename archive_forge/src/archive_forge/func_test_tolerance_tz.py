import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_tolerance_tz(self, unit):
    left = pd.DataFrame({'date': pd.date_range(start=to_datetime('2016-01-02'), freq='D', periods=5, tz=pytz.timezone('UTC'), unit=unit), 'value1': np.arange(5)})
    right = pd.DataFrame({'date': pd.date_range(start=to_datetime('2016-01-01'), freq='D', periods=5, tz=pytz.timezone('UTC'), unit=unit), 'value2': list('ABCDE')})
    result = merge_asof(left, right, on='date', tolerance=Timedelta('1 day'))
    expected = pd.DataFrame({'date': pd.date_range(start=to_datetime('2016-01-02'), freq='D', periods=5, tz=pytz.timezone('UTC'), unit=unit), 'value1': np.arange(5), 'value2': list('BCDEE')})
    tm.assert_frame_equal(result, expected)