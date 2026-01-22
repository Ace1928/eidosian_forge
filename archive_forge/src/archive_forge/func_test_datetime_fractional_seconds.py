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
@pytest.mark.parametrize(('key', 'value', 'warn'), [('date_parser', lambda x: pd.to_datetime(x, format='%Y %m %d %H %M %S.%f'), FutureWarning), ('date_format', '%Y %m %d %H %M %S.%f', None)])
def test_datetime_fractional_seconds(all_parsers, key, value, warn):
    parser = all_parsers
    data = 'year,month,day,hour,minute,second,a,b\n2001,01,05,10,00,0.123456,0.0,10.\n2001,01,5,10,0,0.500000,1.,11.\n'
    result = parser.read_csv_check_warnings(warn, "use 'date_format' instead", StringIO(data), header=0, parse_dates={'ymdHMS': [0, 1, 2, 3, 4, 5]}, **{key: value}, raise_on_extra_warnings=False)
    expected = DataFrame([[datetime(2001, 1, 5, 10, 0, 0, microsecond=123456), 0.0, 10.0], [datetime(2001, 1, 5, 10, 0, 0, microsecond=500000), 1.0, 11.0]], columns=['ymdHMS', 'a', 'b'])
    tm.assert_frame_equal(result, expected)