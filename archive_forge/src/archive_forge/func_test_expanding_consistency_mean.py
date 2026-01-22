import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
def test_expanding_consistency_mean(all_data, min_periods):
    result = all_data.expanding(min_periods=min_periods).mean()
    expected = all_data.expanding(min_periods=min_periods).sum() / all_data.expanding(min_periods=min_periods).count()
    tm.assert_equal(result, expected.astype('float64'))