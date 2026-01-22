import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_count_with_min_periods_exceeding_series_length(frame_or_series):
    result = frame_or_series(range(5)).expanding(min_periods=6).count()
    expected = frame_or_series([np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_equal(result, expected)