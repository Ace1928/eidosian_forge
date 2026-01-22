from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_multiindex_columns_no_data(all_parsers):
    parser = all_parsers
    result = parser.read_csv(StringIO('a0,a1,a2\nb0,b1,b2\n'), header=[0, 1])
    expected = DataFrame([], columns=MultiIndex.from_arrays([['a0', 'a1', 'a2'], ['b0', 'b1', 'b2']]))
    tm.assert_frame_equal(result, expected)