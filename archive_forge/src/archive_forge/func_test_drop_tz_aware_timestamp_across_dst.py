import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_tz_aware_timestamp_across_dst(self, frame_or_series):
    start = Timestamp('2017-10-29', tz='Europe/Berlin')
    end = Timestamp('2017-10-29 04:00:00', tz='Europe/Berlin')
    index = pd.date_range(start, end, freq='15min')
    data = frame_or_series(data=[1] * len(index), index=index)
    result = data.drop(start)
    expected_start = Timestamp('2017-10-29 00:15:00', tz='Europe/Berlin')
    expected_idx = pd.date_range(expected_start, end, freq='15min')
    expected = frame_or_series(data=[1] * len(expected_idx), index=expected_idx)
    tm.assert_equal(result, expected)