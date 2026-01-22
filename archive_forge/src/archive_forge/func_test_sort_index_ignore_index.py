import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('original_dict, sorted_dict, ascending, ignore_index, output_index', [({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, True, [0, 1, 2]), ({'A': [1, 2, 3]}, {'A': [2, 3, 1]}, False, False, [5, 3, 2]), ({'A': [1, 2, 3]}, {'A': [1, 3, 2]}, True, False, [2, 3, 5])])
def test_sort_index_ignore_index(self, inplace, original_dict, sorted_dict, ascending, ignore_index, output_index):
    original_index = [2, 5, 3]
    df = DataFrame(original_dict, index=original_index)
    expected_df = DataFrame(sorted_dict, index=output_index)
    kwargs = {'ascending': ascending, 'ignore_index': ignore_index, 'inplace': inplace}
    if inplace:
        result_df = df.copy()
        result_df.sort_index(**kwargs)
    else:
        result_df = df.sort_index(**kwargs)
    tm.assert_frame_equal(result_df, expected_df)
    tm.assert_frame_equal(df, DataFrame(original_dict, index=original_index))