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
@pytest.mark.parametrize('parse_dates', [[0, 2], ['a', 'c']])
def test_parse_dates_column_list(all_parsers, parse_dates):
    data = 'a,b,c\n01/01/2010,1,15/02/2010'
    parser = all_parsers
    expected = DataFrame({'a': [datetime(2010, 1, 1)], 'b': [1], 'c': [datetime(2010, 2, 15)]})
    expected = expected.set_index(['a', 'b'])
    result = parser.read_csv(StringIO(data), index_col=[0, 1], parse_dates=parse_dates, dayfirst=True)
    tm.assert_frame_equal(result, expected)