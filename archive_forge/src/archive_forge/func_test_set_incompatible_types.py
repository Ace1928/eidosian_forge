import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:'<' not supported between:RuntimeWarning")
@pytest.mark.parametrize('op_name', ['union', 'intersection', 'difference', 'symmetric_difference'])
def test_set_incompatible_types(self, closed, op_name, sort):
    index = monotonic_index(0, 11, closed=closed)
    set_op = getattr(index, op_name)
    if op_name == 'difference':
        expected = index
    else:
        expected = getattr(index.astype('O'), op_name)(Index([1, 2, 3]))
    result = set_op(Index([1, 2, 3]), sort=sort)
    tm.assert_index_equal(result, expected)
    for other_closed in {'right', 'left', 'both', 'neither'} - {closed}:
        other = monotonic_index(0, 11, closed=other_closed)
        expected = getattr(index.astype(object), op_name)(other, sort=sort)
        if op_name == 'difference':
            expected = index
        result = set_op(other, sort=sort)
        tm.assert_index_equal(result, expected)
    other = interval_range(Timestamp('20180101'), periods=9, closed=closed)
    expected = getattr(index.astype(object), op_name)(other, sort=sort)
    if op_name == 'difference':
        expected = index
    result = set_op(other, sort=sort)
    tm.assert_index_equal(result, expected)