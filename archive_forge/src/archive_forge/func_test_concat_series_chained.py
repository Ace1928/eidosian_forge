import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_concat_series_chained(using_copy_on_write):
    ser1 = Series([1, 2, 3], name='a')
    ser2 = Series([4, 5, 6], name='c')
    ser3 = Series([4, 5, 6], name='d')
    result = concat([concat([ser1, ser2], axis=1), ser3], axis=1)
    expected = result.copy()
    if using_copy_on_write:
        assert np.shares_memory(get_array(result, 'a'), get_array(ser1, 'a'))
        assert np.shares_memory(get_array(result, 'c'), get_array(ser2, 'c'))
        assert np.shares_memory(get_array(result, 'd'), get_array(ser3, 'd'))
    else:
        assert not np.shares_memory(get_array(result, 'a'), get_array(ser1, 'a'))
        assert not np.shares_memory(get_array(result, 'c'), get_array(ser2, 'c'))
        assert not np.shares_memory(get_array(result, 'd'), get_array(ser3, 'd'))
    ser1.iloc[0] = 100
    if using_copy_on_write:
        assert not np.shares_memory(get_array(result, 'a'), get_array(ser1, 'a'))
    tm.assert_frame_equal(result, expected)