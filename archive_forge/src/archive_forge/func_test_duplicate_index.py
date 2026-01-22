import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('input_dict, input_index, expected_dict, expected_index', [({'col1': [[1, 2], [3, 4]], 'col2': ['foo', 'bar']}, [0, 0], {'col1': [1, 2, 3, 4], 'col2': ['foo', 'foo', 'bar', 'bar']}, [0, 0, 0, 0]), ({'col1': [[1, 2], [3, 4]], 'col2': ['foo', 'bar']}, pd.Index([0, 0], name='my_index'), {'col1': [1, 2, 3, 4], 'col2': ['foo', 'foo', 'bar', 'bar']}, pd.Index([0, 0, 0, 0], name='my_index')), ({'col1': [[1, 2], [3, 4]], 'col2': ['foo', 'bar']}, pd.MultiIndex.from_arrays([[0, 0], [1, 1]], names=['my_first_index', 'my_second_index']), {'col1': [1, 2, 3, 4], 'col2': ['foo', 'foo', 'bar', 'bar']}, pd.MultiIndex.from_arrays([[0, 0, 0, 0], [1, 1, 1, 1]], names=['my_first_index', 'my_second_index'])), ({'col1': [[1, 2], [3, 4]], 'col2': ['foo', 'bar']}, pd.MultiIndex.from_arrays([[0, 0], [1, 1]], names=['my_index', None]), {'col1': [1, 2, 3, 4], 'col2': ['foo', 'foo', 'bar', 'bar']}, pd.MultiIndex.from_arrays([[0, 0, 0, 0], [1, 1, 1, 1]], names=['my_index', None]))])
def test_duplicate_index(input_dict, input_index, expected_dict, expected_index):
    df = pd.DataFrame(input_dict, index=input_index, dtype=object)
    result = df.explode('col1')
    expected = pd.DataFrame(expected_dict, index=expected_index, dtype=object)
    tm.assert_frame_equal(result, expected)