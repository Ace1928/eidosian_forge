from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_index(all_parsers):
    data = 'x,y'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=0)
    expected = DataFrame(columns=['y'], index=Index([], name='x'))
    tm.assert_frame_equal(result, expected)