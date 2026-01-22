from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
def test_read_duplicate_index_explicit(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo,12,13,14,15\nbar,12,13,14,15\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)
    expected = DataFrame([[2, 3, 4, 5], [7, 8, 9, 10], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15], [12, 13, 14, 15]], columns=['A', 'B', 'C', 'D'], index=Index(['foo', 'bar', 'baz', 'qux', 'foo', 'bar'], name='index'))
    tm.assert_frame_equal(result, expected)