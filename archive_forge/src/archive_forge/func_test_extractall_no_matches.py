from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
@pytest.mark.parametrize('data, names', [([], (None,)), ([], ('i1',)), ([], (None, 'i2')), ([], ('i1', 'i2')), (['a3', 'b3', 'd4c2'], (None,)), (['a3', 'b3', 'd4c2'], ('i1', 'i2')), (['a3', 'b3', 'd4c2'], (None, 'i2')), (['a3', 'b3', 'd4c2'], ('i1', 'i2'))])
def test_extractall_no_matches(data, names, any_string_dtype):
    n = len(data)
    if len(names) == 1:
        index = Index(range(n), name=names[0])
    else:
        tuples = (tuple([i] * (n - 1)) for i in range(n))
        index = MultiIndex.from_tuples(tuples, names=names)
    s = Series(data, name='series_name', index=index, dtype=any_string_dtype)
    expected_index = MultiIndex.from_tuples([], names=names + ('match',))
    result = s.str.extractall('(z)')
    expected = DataFrame(columns=[0], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extractall('(z)(z)')
    expected = DataFrame(columns=[0, 1], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extractall('(?P<first>z)')
    expected = DataFrame(columns=['first'], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extractall('(?P<first>z)(?P<second>z)')
    expected = DataFrame(columns=['first', 'second'], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)
    result = s.str.extractall('(z)(?P<second>z)')
    expected = DataFrame(columns=[0, 'second'], index=expected_index, dtype=any_string_dtype)
    tm.assert_frame_equal(result, expected)