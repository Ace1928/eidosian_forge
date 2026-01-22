import pytest
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm
def test_tdi_shift_minutes(self):
    idx = TimedeltaIndex(['5 hours', '6 hours', '9 hours'], name='xxx')
    tm.assert_index_equal(idx.shift(0, freq='min'), idx)
    exp = TimedeltaIndex(['05:03:00', '06:03:00', '9:03:00'], name='xxx')
    tm.assert_index_equal(idx.shift(3, freq='min'), exp)
    exp = TimedeltaIndex(['04:57:00', '05:57:00', '8:57:00'], name='xxx')
    tm.assert_index_equal(idx.shift(-3, freq='min'), exp)