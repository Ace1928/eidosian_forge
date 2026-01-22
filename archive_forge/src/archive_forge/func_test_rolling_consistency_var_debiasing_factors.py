import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_rolling_consistency_var_debiasing_factors(all_data, rolling_consistency_cases, center):
    window, min_periods = rolling_consistency_cases
    var_unbiased_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var()
    var_biased_x = all_data.rolling(window=window, min_periods=min_periods, center=center).var(ddof=0)
    var_debiasing_factors_x = all_data.rolling(window=window, min_periods=min_periods, center=center).count().divide((all_data.rolling(window=window, min_periods=min_periods, center=center).count() - 1.0).replace(0.0, np.nan))
    tm.assert_equal(var_unbiased_x, var_biased_x * var_debiasing_factors_x)