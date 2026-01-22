from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_dtype_all_columns_empty(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO('A,B'), dtype=str)
    expected = DataFrame({'A': [], 'B': []}, dtype=str)
    tm.assert_frame_equal(result, expected)