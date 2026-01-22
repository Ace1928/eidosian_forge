from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_ffill_mixed_dtypes_without_missing_data(self):
    series = Series([datetime(2015, 1, 1, tzinfo=pytz.utc), 1])
    result = series.ffill()
    tm.assert_series_equal(series, result)