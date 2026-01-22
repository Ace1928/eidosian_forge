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
def test_multiple_date_cols_int_cast(all_parsers):
    data = 'KORD,19990127, 19:00:00, 18:56:00, 0.8100\nKORD,19990127, 20:00:00, 19:56:00, 0.0100\nKORD,19990127, 21:00:00, 20:56:00, -0.5900\nKORD,19990127, 21:00:00, 21:18:00, -0.9900\nKORD,19990127, 22:00:00, 21:56:00, -0.5900\nKORD,19990127, 23:00:00, 22:56:00, -0.5900'
    parse_dates = {'actual': [1, 2], 'nominal': [1, 3]}
    parser = all_parsers
    kwds = {'header': None, 'parse_dates': parse_dates, 'date_parser': pd.to_datetime}
    result = parser.read_csv_check_warnings(FutureWarning, "use 'date_format' instead", StringIO(data), **kwds, raise_on_extra_warnings=False)
    expected = DataFrame([[datetime(1999, 1, 27, 19, 0), datetime(1999, 1, 27, 18, 56), 'KORD', 0.81], [datetime(1999, 1, 27, 20, 0), datetime(1999, 1, 27, 19, 56), 'KORD', 0.01], [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 20, 56), 'KORD', -0.59], [datetime(1999, 1, 27, 21, 0), datetime(1999, 1, 27, 21, 18), 'KORD', -0.99], [datetime(1999, 1, 27, 22, 0), datetime(1999, 1, 27, 21, 56), 'KORD', -0.59], [datetime(1999, 1, 27, 23, 0), datetime(1999, 1, 27, 22, 56), 'KORD', -0.59]], columns=['actual', 'nominal', 0, 4])
    result = result[expected.columns]
    tm.assert_frame_equal(result, expected)