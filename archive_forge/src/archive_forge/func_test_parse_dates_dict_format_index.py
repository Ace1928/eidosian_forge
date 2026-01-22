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
def test_parse_dates_dict_format_index(all_parsers):
    parser = all_parsers
    data = 'a,b\n2019-12-31,31-12-2019\n2020-12-31,31-12-2020'
    result = parser.read_csv(StringIO(data), date_format={'a': '%Y-%m-%d'}, parse_dates=True, index_col=0)
    expected = DataFrame({'b': ['31-12-2019', '31-12-2020']}, index=Index([Timestamp('2019-12-31'), Timestamp('2020-12-31')], name='a'))
    tm.assert_frame_equal(result, expected)