from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_min_periods_count(series, step):
    result = series.rolling(len(series) + 1, min_periods=0, step=step).count()
    expected = series.rolling(len(series), min_periods=0, step=step).count()
    nan_mask = isna(result)
    tm.assert_series_equal(nan_mask, isna(expected))
    nan_mask = ~nan_mask
    tm.assert_almost_equal(result[nan_mask], expected[nan_mask])