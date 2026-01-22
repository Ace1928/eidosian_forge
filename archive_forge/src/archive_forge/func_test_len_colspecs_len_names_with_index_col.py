from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
@pytest.mark.parametrize('colspecs, names, widths, index_col, expected', [([(0, 6), (6, 12), (12, 18), (18, None)], list('abc'), None, 0, DataFrame(index=['col1', 'ba'], columns=['a', 'b', 'c'], data=[['col2', 'col3', 'col4'], ['b   ba', '2', np.nan]])), ([(0, 6), (6, 12), (12, 18), (18, None)], list('ab'), None, [0, 1], DataFrame(index=[['col1', 'ba'], ['col2', 'b   ba']], columns=['a', 'b'], data=[['col3', 'col4'], ['2', np.nan]])), ([(0, 6), (6, 12), (12, 18), (18, None)], list('a'), None, [0, 1, 2], DataFrame(index=[['col1', 'ba'], ['col2', 'b   ba'], ['col3', '2']], columns=['a'], data=[['col4'], [np.nan]])), (None, list('abc'), [6] * 4, 0, DataFrame(index=['col1', 'ba'], columns=['a', 'b', 'c'], data=[['col2', 'col3', 'col4'], ['b   ba', '2', np.nan]])), (None, list('ab'), [6] * 4, [0, 1], DataFrame(index=[['col1', 'ba'], ['col2', 'b   ba']], columns=['a', 'b'], data=[['col3', 'col4'], ['2', np.nan]])), (None, list('a'), [6] * 4, [0, 1, 2], DataFrame(index=[['col1', 'ba'], ['col2', 'b   ba'], ['col3', '2']], columns=['a'], data=[['col4'], [np.nan]]))])
def test_len_colspecs_len_names_with_index_col(colspecs, names, widths, index_col, expected):
    data = 'col1  col2  col3  col4\n    bab   ba    2'
    result = read_fwf(StringIO(data), colspecs=colspecs, names=names, widths=widths, index_col=index_col)
    tm.assert_frame_equal(result, expected)