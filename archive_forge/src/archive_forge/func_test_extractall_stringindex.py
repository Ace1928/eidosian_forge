from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extractall_stringindex(any_string_dtype):
    s = Series(['a1a2', 'b1', 'c1'], name='xxx', dtype=any_string_dtype)
    result = s.str.extractall('[ab](?P<digit>\\d)')
    expected = DataFrame({'digit': ['1', '2', '1']}, index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=[None, 'match']), dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    if any_string_dtype == 'object':
        for idx in [Index(['a1a2', 'b1', 'c1'], dtype=object), Index(['a1a2', 'b1', 'c1'], name='xxx', dtype=object)]:
            result = idx.str.extractall('[ab](?P<digit>\\d)')
            tm.assert_frame_equal(result, expected)
    s = Series(['a1a2', 'b1', 'c1'], name='s_name', index=Index(['XX', 'yy', 'zz'], name='idx_name'), dtype=any_string_dtype)
    result = s.str.extractall('[ab](?P<digit>\\d)')
    expected = DataFrame({'digit': ['1', '2', '1']}, index=MultiIndex.from_tuples([('XX', 0), ('XX', 1), ('yy', 0)], names=['idx_name', 'match']), dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)