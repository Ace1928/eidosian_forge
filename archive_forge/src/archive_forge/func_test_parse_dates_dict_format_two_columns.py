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
@pytest.mark.parametrize('key, parse_dates', [('a_b', [[0, 1]]), ('foo', {'foo': [0, 1]})])
def test_parse_dates_dict_format_two_columns(all_parsers, key, parse_dates):
    parser = all_parsers
    data = 'a,b\n31-,12-2019\n31-,12-2020'
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with tm.assert_produces_warning((FutureWarning, DeprecationWarning), match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), date_format={key: '%d- %m-%Y'}, parse_dates=parse_dates)
    expected = DataFrame({key: [Timestamp('2019-12-31'), Timestamp('2020-12-31')]})
    tm.assert_frame_equal(result, expected)