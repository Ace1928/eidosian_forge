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
def test_from_csv_with_mixed_offsets(all_parsers):
    parser = all_parsers
    data = 'a\n2020-01-01T00:00:00+01:00\n2020-01-01T00:00:00+00:00'
    result = parser.read_csv(StringIO(data), parse_dates=['a'])['a']
    expected = Series([Timestamp('2020-01-01 00:00:00+01:00'), Timestamp('2020-01-01 00:00:00+00:00')], name='a', index=[0, 1])
    tm.assert_series_equal(result, expected)