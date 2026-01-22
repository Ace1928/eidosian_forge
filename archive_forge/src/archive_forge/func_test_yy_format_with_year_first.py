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
@pytest.mark.xfail(reason='yearfirst is not surfaced in read_*')
@pytest.mark.parametrize('parse_dates', [[['date', 'time']], [[0, 1]]])
def test_yy_format_with_year_first(all_parsers, parse_dates):
    data = 'date,time,B,C\n090131,0010,1,2\n090228,1020,3,4\n090331,0830,5,6\n'
    parser = all_parsers
    result = parser.read_csv_check_warnings(UserWarning, 'Could not infer format', StringIO(data), index_col=0, parse_dates=parse_dates)
    index = DatetimeIndex([datetime(2009, 1, 31, 0, 10, 0), datetime(2009, 2, 28, 10, 20, 0), datetime(2009, 3, 31, 8, 30, 0)], dtype=object, name='date_time')
    expected = DataFrame({'B': [1, 3, 5], 'C': [2, 4, 6]}, index=index)
    tm.assert_frame_equal(result, expected)