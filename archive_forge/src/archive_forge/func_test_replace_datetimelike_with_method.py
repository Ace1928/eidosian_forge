import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('as_period', [True, False])
@pytest.mark.parametrize('as_categorical', [True, False])
def test_replace_datetimelike_with_method(self, as_period, as_categorical):
    idx = pd.date_range('2016-01-01', periods=5, tz='US/Pacific')
    if as_period:
        idx = idx.tz_localize(None).to_period('D')
    ser = pd.Series(idx)
    ser.iloc[-2] = pd.NaT
    if as_categorical:
        ser = ser.astype('category')
    self._check_replace_with_method(ser)