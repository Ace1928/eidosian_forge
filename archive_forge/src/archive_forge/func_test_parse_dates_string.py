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
def test_parse_dates_string(all_parsers):
    data = 'date,A,B,C\n20090101,a,1,2\n20090102,b,3,4\n20090103,c,4,5\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col='date', parse_dates=['date'])
    index = date_range('1/1/2009', periods=3, name='date')._with_freq(None)
    expected = DataFrame({'A': ['a', 'b', 'c'], 'B': [1, 3, 4], 'C': [2, 4, 5]}, index=index)
    tm.assert_frame_equal(result, expected)