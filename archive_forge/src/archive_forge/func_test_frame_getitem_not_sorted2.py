import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key', [None, lambda x: x])
def test_frame_getitem_not_sorted2(self, key):
    df = DataFrame({'col1': ['b', 'd', 'b', 'a'], 'col2': [3, 1, 1, 2], 'data': ['one', 'two', 'three', 'four']})
    df2 = df.set_index(['col1', 'col2'])
    df2_original = df2.copy()
    df2.index = df2.index.set_levels(['b', 'd', 'a'], level='col1')
    df2.index = df2.index.set_codes([0, 1, 0, 2], level='col1')
    assert not df2.index.is_monotonic_increasing
    assert df2_original.index.equals(df2.index)
    expected = df2.sort_index(key=key)
    assert expected.index.is_monotonic_increasing
    result = df2.sort_index(level=0, key=key)
    assert result.index.is_monotonic_increasing
    tm.assert_frame_equal(result, expected)