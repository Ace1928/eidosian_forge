import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_rolling_consistency_constant(consistent_data, rolling_consistency_cases, center):
    window, min_periods = rolling_consistency_cases
    count_x = consistent_data.rolling(window=window, min_periods=min_periods, center=center).count()
    mean_x = consistent_data.rolling(window=window, min_periods=min_periods, center=center).mean()
    corr_x_x = consistent_data.rolling(window=window, min_periods=min_periods, center=center).corr(consistent_data)
    exp = consistent_data.max() if isinstance(consistent_data, Series) else consistent_data.max().max()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = exp
    tm.assert_equal(mean_x, expected)
    expected[:] = np.nan
    tm.assert_equal(corr_x_x, expected)