from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_bmonth(self, unit):
    rng = date_range(START, END, freq=pd.offsets.BMonthEnd(), unit=unit)
    shifted = rng.shift(1, freq=pd.offsets.BDay())
    assert shifted[0] == rng[0] + pd.offsets.BDay()
    rng = date_range(START, END, freq=pd.offsets.BMonthEnd(), unit=unit)
    with tm.assert_produces_warning(pd.errors.PerformanceWarning):
        shifted = rng.shift(1, freq=pd.offsets.CDay())
        assert shifted[0] == rng[0] + pd.offsets.CDay()