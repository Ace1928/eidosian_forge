from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_dataframe_capture_groups_index(index, any_string_dtype):
    data = ['A1', 'B2', 'C']
    if len(index) < len(data):
        pytest.skip(f'Index needs more than {len(data)} values')
    index = index[:len(data)]
    s = Series(data, index=index, dtype=any_string_dtype)
    result = s.str.extract('(\\d)', expand=True)
    expected = DataFrame(['1', '2', np.nan], index=index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extract('(?P<letter>\\D)(?P<number>\\d)?', expand=True)
    expected = DataFrame([['A', '1'], ['B', '2'], ['C', np.nan]], columns=['letter', 'number'], index=index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)