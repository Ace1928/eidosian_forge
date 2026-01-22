from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_multi_index(all_parsers):
    data = 'x,y,z'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=['x', 'y'])
    expected = DataFrame(columns=['z'], index=MultiIndex.from_arrays([[]] * 2, names=['x', 'y']))
    tm.assert_frame_equal(result, expected)