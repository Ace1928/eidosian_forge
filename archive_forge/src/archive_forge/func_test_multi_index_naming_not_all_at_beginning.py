from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_multi_index_naming_not_all_at_beginning(all_parsers):
    parser = all_parsers
    data = ',Unnamed: 2,\na,c,1\na,d,2\nb,c,3\nb,d,4'
    result = parser.read_csv(StringIO(data), index_col=[0, 2])
    expected = DataFrame({'Unnamed: 2': ['c', 'd', 'c', 'd']}, index=MultiIndex(levels=[['a', 'b'], [1, 2, 3, 4]], codes=[[0, 0, 1, 1], [0, 1, 2, 3]]))
    tm.assert_frame_equal(result, expected)