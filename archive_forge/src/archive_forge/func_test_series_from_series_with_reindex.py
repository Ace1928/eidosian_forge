import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_series_from_series_with_reindex(using_copy_on_write, warn_copy_on_write):
    ser = Series([1, 2, 3], name='name')
    for index in [ser.index, ser.index.copy(), list(ser.index), ser.index.rename('idx')]:
        result = Series(ser, index=index)
        assert np.shares_memory(ser.values, result.values)
        with tm.assert_cow_warning(warn_copy_on_write):
            result.iloc[0] = 0
        if using_copy_on_write:
            assert ser.iloc[0] == 1
        else:
            assert ser.iloc[0] == 0
    result = Series(ser, index=[0, 1, 2, 3])
    assert not np.shares_memory(ser.values, result.values)
    if using_copy_on_write:
        assert not result._mgr.blocks[0].refs.has_reference()