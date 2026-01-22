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
def test_parse_dates_and_string_dtype(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2019-12-31\n'
    result = parser.read_csv(StringIO(data), dtype='string', parse_dates=['b'])
    expected = DataFrame({'a': ['1'], 'b': [Timestamp('2019-12-31')]})
    expected['a'] = expected['a'].astype('string')
    tm.assert_frame_equal(result, expected)