from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_shift_tzaware(self, tz_naive_fixture, unit):
    tz = tz_naive_fixture
    idx = DatetimeIndex([], name='xxx', tz=tz).as_unit(unit)
    tm.assert_index_equal(idx.shift(0, freq='h'), idx)
    tm.assert_index_equal(idx.shift(3, freq='h'), idx)
    idx = DatetimeIndex(['2011-01-01 10:00', '2011-01-01 11:00', '2011-01-01 12:00'], name='xxx', tz=tz, freq='h').as_unit(unit)
    tm.assert_index_equal(idx.shift(0, freq='h'), idx)
    exp = DatetimeIndex(['2011-01-01 13:00', '2011-01-01 14:00', '2011-01-01 15:00'], name='xxx', tz=tz, freq='h').as_unit(unit)
    tm.assert_index_equal(idx.shift(3, freq='h'), exp)
    exp = DatetimeIndex(['2011-01-01 07:00', '2011-01-01 08:00', '2011-01-01 09:00'], name='xxx', tz=tz, freq='h').as_unit(unit)
    tm.assert_index_equal(idx.shift(-3, freq='h'), exp)