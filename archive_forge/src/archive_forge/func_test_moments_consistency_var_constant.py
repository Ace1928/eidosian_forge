import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('ddof', [0, 1])
def test_moments_consistency_var_constant(consistent_data, rolling_consistency_cases, center, ddof):
    window, min_periods = rolling_consistency_cases
    count_x = consistent_data.rolling(window=window, min_periods=min_periods, center=center).count()
    var_x = consistent_data.rolling(window=window, min_periods=min_periods, center=center).var(ddof=ddof)
    assert not (var_x > 0).any().any()
    expected = consistent_data * np.nan
    expected[count_x >= max(min_periods, 1)] = 0.0
    if ddof == 1:
        expected[count_x < 2] = np.nan
    tm.assert_equal(var_x, expected)