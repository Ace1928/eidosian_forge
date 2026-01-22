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
def test_parse_dates_empty_string(all_parsers):
    parser = all_parsers
    data = 'Date,test\n2012-01-01,1\n,2'
    result = parser.read_csv(StringIO(data), parse_dates=['Date'], na_filter=False)
    expected = DataFrame([[datetime(2012, 1, 1), 1], [pd.NaT, 2]], columns=['Date', 'test'])
    tm.assert_frame_equal(result, expected)