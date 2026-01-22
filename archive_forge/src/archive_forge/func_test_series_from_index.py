import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('idx', [Index([1, 2]), DatetimeIndex([Timestamp('2019-12-31'), Timestamp('2020-12-31')]), PeriodIndex([Period('2019-12-31'), Period('2020-12-31')]), TimedeltaIndex([Timedelta('1 days'), Timedelta('2 days')])])
def test_series_from_index(using_copy_on_write, idx):
    ser = Series(idx)
    expected = idx.copy(deep=True)
    if using_copy_on_write:
        assert np.shares_memory(get_array(ser), get_array(idx))
        assert not ser._mgr._has_no_reference(0)
    else:
        assert not np.shares_memory(get_array(ser), get_array(idx))
    ser.iloc[0] = ser.iloc[1]
    tm.assert_index_equal(idx, expected)