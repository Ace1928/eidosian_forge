import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unstack_idx, expected_values, expected_index, expected_columns', [(('A', 'a'), [[1, 1], [1, 1], [1, 1], [1, 1]], MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=['B', 'C']), MultiIndex.from_tuples([('a',), ('b',)], names=[('A', 'a')])), ((('A', 'a'), 'B'), [[1, 1, 1, 1], [1, 1, 1, 1]], Index([3, 4], name='C'), MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)], names=[('A', 'a'), 'B']))])
def test_unstack_mixed_type_name_in_multiindex(unstack_idx, expected_values, expected_index, expected_columns):
    idx = MultiIndex.from_product([['a', 'b'], [1, 2], [3, 4]], names=[('A', 'a'), 'B', 'C'])
    ser = Series(1, index=idx)
    result = ser.unstack(unstack_idx)
    expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
    tm.assert_frame_equal(result, expected)