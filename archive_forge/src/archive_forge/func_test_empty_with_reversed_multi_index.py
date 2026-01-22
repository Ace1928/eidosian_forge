from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_reversed_multi_index(all_parsers):
    data = 'x,y,z'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=[1, 0])
    expected = DataFrame(columns=['z'], index=MultiIndex.from_arrays([[]] * 2, names=['y', 'x']))
    tm.assert_frame_equal(result, expected)