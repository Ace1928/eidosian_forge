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
def test_parse_dates_arrow_engine(all_parsers):
    parser = all_parsers
    data = 'a,b\n2000-01-01 00:00:00,1\n2000-01-01 00:00:01,1'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])
    if parser.engine == 'pyarrow':
        result['a'] = result['a'].dt.as_unit('ns')
    expected = DataFrame({'a': [Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 00:00:01')], 'b': 1})
    tm.assert_frame_equal(result, expected)