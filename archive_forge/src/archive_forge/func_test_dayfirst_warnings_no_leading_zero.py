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
@pytest.mark.parametrize('date_string, dayfirst', [pytest.param('31/1/2014', False, id='second date is single-digit'), pytest.param('1/31/2014', True, id='first date is single-digit')])
def test_dayfirst_warnings_no_leading_zero(date_string, dayfirst):
    initial_value = f'date\n{date_string}'
    expected = DatetimeIndex(['2014-01-31'], dtype='datetime64[ns]', freq=None, name='date')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    with tm.assert_produces_warning(UserWarning, match=warning_msg):
        res = read_csv(StringIO(initial_value), parse_dates=['date'], index_col='date', dayfirst=dayfirst).index
    tm.assert_index_equal(expected, res)