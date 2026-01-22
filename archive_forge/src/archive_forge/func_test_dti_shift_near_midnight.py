from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('shift, result_time', [[0, '2014-11-14 00:00:00'], [-1, '2014-11-13 23:00:00'], [1, '2014-11-14 01:00:00']])
def test_dti_shift_near_midnight(self, shift, result_time, unit):
    dt = datetime(2014, 11, 14, 0)
    dt_est = pytz.timezone('EST').localize(dt)
    idx = DatetimeIndex([dt_est]).as_unit(unit)
    ser = Series(data=[1], index=idx)
    result = ser.shift(shift, freq='h')
    exp_index = DatetimeIndex([result_time], tz='EST').as_unit(unit)
    expected = Series(1, index=exp_index)
    tm.assert_series_equal(result, expected)