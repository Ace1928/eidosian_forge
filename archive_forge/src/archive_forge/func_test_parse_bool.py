from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,kwargs,expected', [('A,B\nTrue,1\nFalse,2\nTrue,3', {}, DataFrame([[True, 1], [False, 2], [True, 3]], columns=['A', 'B'])), ('A,B\nYES,1\nno,2\nyes,3\nNo,3\nYes,3', {'true_values': ['yes', 'Yes', 'YES'], 'false_values': ['no', 'NO', 'No']}, DataFrame([[True, 1], [False, 2], [True, 3], [False, 3], [True, 3]], columns=['A', 'B'])), ('A,B\nTRUE,1\nFALSE,2\nTRUE,3', {}, DataFrame([[True, 1], [False, 2], [True, 3]], columns=['A', 'B'])), ('A,B\nfoo,bar\nbar,foo', {'true_values': ['foo'], 'false_values': ['bar']}, DataFrame([[True, False], [False, True]], columns=['A', 'B']))])
def test_parse_bool(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)