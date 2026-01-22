import pytest
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import TimedeltaIndex
import pandas._testing as tm
def test_tdi_shift_empty(self):
    idx = TimedeltaIndex([], name='xxx')
    tm.assert_index_equal(idx.shift(0, freq='h'), idx)
    tm.assert_index_equal(idx.shift(3, freq='h'), idx)