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
@pytest.mark.parametrize('parse_dates,index_col', [({'nominal': [1, 2]}, 'nominal'), ({'nominal': [1, 2]}, 0), ([[1, 2]], 0)])
def test_multiple_date_cols_index(all_parsers, parse_dates, index_col):
    parser = all_parsers
    data = '\nID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir\nKORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000\nKORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000\nKORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000\nKORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000\nKORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000\nKORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000\n'
    expected = DataFrame([[datetime(1999, 1, 27, 19, 0), 'KORD1', ' 18:56:00', 0.81, 2.81, 7.2, 0.0, 280.0], [datetime(1999, 1, 27, 20, 0), 'KORD2', ' 19:56:00', 0.01, 2.21, 7.2, 0.0, 260.0], [datetime(1999, 1, 27, 21, 0), 'KORD3', ' 20:56:00', -0.59, 2.21, 5.7, 0.0, 280.0], [datetime(1999, 1, 27, 21, 0), 'KORD4', ' 21:18:00', -0.99, 2.01, 3.6, 0.0, 270.0], [datetime(1999, 1, 27, 22, 0), 'KORD5', ' 21:56:00', -0.59, 1.71, 5.1, 0.0, 290.0], [datetime(1999, 1, 27, 23, 0), 'KORD6', ' 22:56:00', -0.59, 1.71, 4.6, 0.0, 280.0]], columns=['nominal', 'ID', 'ActualTime', 'TDew', 'TAir', 'Windspeed', 'Precip', 'WindDir'])
    expected = expected.set_index('nominal')
    if not isinstance(parse_dates, dict):
        expected.index.name = 'date_NominalTime'
    depr_msg = "Support for nested sequences for 'parse_dates' in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), parse_dates=parse_dates, index_col=index_col)
    tm.assert_frame_equal(result, expected)