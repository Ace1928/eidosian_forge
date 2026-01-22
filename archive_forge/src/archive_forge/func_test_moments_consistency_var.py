import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('ddof', [0, 1])
def test_moments_consistency_var(all_data, rolling_consistency_cases, center, ddof):
    window, min_periods = rolling_consistency_cases
    var_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(ddof=ddof)
    assert not (var_x < 0).any().any()
    if ddof == 0:
        mean_x = all_data.rolling(window=window, min_periods=min_periods, center=center).mean()
        mean_x2 = (all_data * all_data).rolling(window=window, min_periods=min_periods, center=center).mean()
        tm.assert_equal(var_x, mean_x2 - mean_x * mean_x)