from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('sort, ascending, normalize, name, expected_data, expected_index', [(False, None, False, 'count', [1, 2, 1], [(1, 1, 1), (2, 4, 6), (2, 0, 0)]), (True, True, False, 'count', [1, 1, 2], [(1, 1, 1), (2, 6, 4), (2, 0, 0)]), (True, False, False, 'count', [2, 1, 1], [(1, 1, 1), (4, 2, 6), (0, 2, 0)]), (True, False, True, 'proportion', [0.5, 0.25, 0.25], [(1, 1, 1), (4, 2, 6), (0, 2, 0)])])
def test_data_frame_value_counts(animals_df, sort, ascending, normalize, name, expected_data, expected_index):
    result_frame = animals_df.value_counts(sort=sort, ascending=ascending, normalize=normalize)
    expected = Series(data=expected_data, index=MultiIndex.from_arrays(expected_index, names=['key', 'num_legs', 'num_wings']), name=name)
    tm.assert_series_equal(result_frame, expected)
    result_frame_groupby = animals_df.groupby('key').value_counts(sort=sort, ascending=ascending, normalize=normalize)
    tm.assert_series_equal(result_frame_groupby, expected)