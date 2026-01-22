import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_quantile(series):
    result = series.expanding().quantile(0.5)
    rolling_result = series.rolling(window=len(series), min_periods=1).quantile(0.5)
    tm.assert_almost_equal(result, rolling_result)