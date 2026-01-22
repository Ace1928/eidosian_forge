import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('expected_data, ignore', [[[10.0, 5.0, 2.5, 11.25], False], [[10.0, 5.0, 5.0, 12.5], True]])
def test_ewm_sum(expected_data, ignore):
    data = Series([10, 0, np.nan, 10])
    result = data.ewm(alpha=0.5, ignore_na=ignore).sum()
    expected = Series(expected_data)
    tm.assert_series_equal(result, expected)