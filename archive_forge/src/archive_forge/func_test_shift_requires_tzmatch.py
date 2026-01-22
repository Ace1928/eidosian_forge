from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_shift_requires_tzmatch(self):
    dti = pd.date_range('2016-01-01', periods=3, tz='UTC')
    dta = dti._data
    fill_value = pd.Timestamp('2020-10-18 18:44', tz='US/Pacific')
    result = dta.shift(1, fill_value=fill_value)
    expected = dta.shift(1, fill_value=fill_value.tz_convert('UTC'))
    tm.assert_equal(result, expected)