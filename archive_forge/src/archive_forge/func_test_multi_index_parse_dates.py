from datetime import (
from io import StringIO
from dateutil.parser import parse as du_parse
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import parsing
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
@xfail_pyarrow
@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers, index_col):
    data = 'index1,index2,A,B,C\n20090101,one,a,1,2\n20090101,two,b,3,4\n20090101,three,c,4,5\n20090102,one,a,1,2\n20090102,two,b,3,4\n20090102,three,c,4,5\n20090103,one,a,1,2\n20090103,two,b,3,4\n20090103,three,c,4,5\n'
    parser = all_parsers
    index = MultiIndex.from_product([(datetime(2009, 1, 1), datetime(2009, 1, 2), datetime(2009, 1, 3)), ('one', 'two', 'three')], names=['index1', 'index2'])
    if index_col == [1, 0]:
        index = index.swaplevel(0, 1)
    expected = DataFrame([['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5], ['a', 1, 2], ['b', 3, 4], ['c', 4, 5]], columns=['A', 'B', 'C'], index=index)
    result = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), index_col=index_col, parse_dates=True)
    tm.assert_frame_equal(result, expected)