from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('origin_dict, output_dict, ignore_index, output_index', [({'A': [2, 2, 3]}, {'A': [2, 3]}, True, [0, 1]), ({'A': [2, 2, 3]}, {'A': [2, 3]}, False, [0, 2]), ({'A': [2, 2, 3], 'B': [2, 2, 4]}, {'A': [2, 3], 'B': [2, 4]}, True, [0, 1]), ({'A': [2, 2, 3], 'B': [2, 2, 4]}, {'A': [2, 3], 'B': [2, 4]}, False, [0, 2])])
def test_drop_duplicates_ignore_index(inplace, origin_dict, output_dict, ignore_index, output_index):
    df = DataFrame(origin_dict)
    expected = DataFrame(output_dict, index=output_index)
    if inplace:
        result_df = df.copy()
        result_df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)
    else:
        result_df = df.drop_duplicates(ignore_index=ignore_index, inplace=inplace)
    tm.assert_frame_equal(result_df, expected)
    tm.assert_frame_equal(df, DataFrame(origin_dict))