import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@td.skip_if_no('pyarrow')
def test_merge_asof_pyarrow_td_tolerance():
    ser = pd.Series([datetime.datetime(2023, 1, 1)], dtype='timestamp[us, UTC][pyarrow]')
    df = pd.DataFrame({'timestamp': ser, 'value': [1]})
    result = merge_asof(df, df, on='timestamp', tolerance=Timedelta('1s'))
    expected = pd.DataFrame({'timestamp': ser, 'value_x': [1], 'value_y': [1]})
    tm.assert_frame_equal(result, expected)