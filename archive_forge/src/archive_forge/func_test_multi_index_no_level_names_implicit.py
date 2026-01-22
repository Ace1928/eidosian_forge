from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_multi_index_no_level_names_implicit(all_parsers):
    parser = all_parsers
    data = 'A,B,C,D\nfoo,one,2,3,4,5\nfoo,two,7,8,9,10\nfoo,three,12,13,14,15\nbar,one,12,13,14,15\nbar,two,12,13,14,15\n'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], columns=['A', 'B', 'C', 'D'], index=MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('foo', 'three'), ('bar', 'one'), ('bar', 'two')]))
    tm.assert_frame_equal(result, expected)