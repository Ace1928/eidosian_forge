import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.filterwarnings('ignore:Setting a value on a view:FutureWarning')
@pytest.mark.parametrize('fastpath', [False, True])
@pytest.mark.parametrize('dtype', [None, 'int64'])
@pytest.mark.parametrize('idx', [None, pd.RangeIndex(start=0, stop=3, step=1)])
def test_series_from_block_manager(using_copy_on_write, idx, dtype, fastpath):
    ser = Series([1, 2, 3], dtype='int64')
    ser_orig = ser.copy()
    msg = "The 'fastpath' keyword in pd.Series is deprecated"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype=dtype, fastpath=fastpath, index=idx)
    assert np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert not ser2._mgr._has_no_reference(0)
    ser2.iloc[0] = 100
    if using_copy_on_write:
        tm.assert_series_equal(ser, ser_orig)
    else:
        expected = Series([100, 2, 3])
        tm.assert_series_equal(ser, expected)