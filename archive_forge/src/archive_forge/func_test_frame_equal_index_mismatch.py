import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('check_like', [True, False])
def test_frame_equal_index_mismatch(check_like, obj_fixture, using_infer_string):
    if using_infer_string:
        dtype = 'string'
    else:
        dtype = 'object'
    msg = f"{obj_fixture}\\.index are different\n\n{obj_fixture}\\.index values are different \\(33\\.33333 %\\)\n\\[left\\]:  Index\\(\\['a', 'b', 'c'\\], dtype='{dtype}'\\)\n\\[right\\]: Index\\(\\['a', 'b', 'd'\\], dtype='{dtype}'\\)\nAt positional index 2, first diff: c != d"
    df1 = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df2 = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'd'])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_like=check_like, obj=obj_fixture)