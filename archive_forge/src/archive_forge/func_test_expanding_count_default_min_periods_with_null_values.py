import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_count_default_min_periods_with_null_values(frame_or_series):
    values = [1, 2, 3, np.nan, 4, 5, 6]
    expected_counts = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    result = frame_or_series(values).expanding().count()
    expected = frame_or_series(expected_counts)
    tm.assert_equal(result, expected)