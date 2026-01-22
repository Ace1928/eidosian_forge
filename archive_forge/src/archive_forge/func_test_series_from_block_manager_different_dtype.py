import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_from_block_manager_different_dtype(using_copy_on_write):
    ser = Series([1, 2, 3], dtype='int64')
    msg = 'Passing a SingleBlockManager to Series'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        ser2 = Series(ser._mgr, dtype='int32')
    assert not np.shares_memory(get_array(ser), get_array(ser2))
    if using_copy_on_write:
        assert ser2._mgr._has_no_reference(0)