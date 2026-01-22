from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_multiindex_columns_not_leading_index_col(all_parsers):
    parser = all_parsers
    data = 'a,b,c,d\ne,f,g,h\nx,y,1,2\n'
    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=1)
    cols = MultiIndex.from_tuples([('a', 'e'), ('c', 'g'), ('d', 'h')], names=['b', 'f'])
    expected = DataFrame([['x', 1, 2]], columns=cols, index=['y'])
    tm.assert_frame_equal(result, expected)