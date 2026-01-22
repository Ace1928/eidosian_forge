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
@skip_pyarrow
@pytest.mark.parametrize('date_string,dayfirst,expected', [('13/02/2019', False, datetime(2019, 2, 13)), ('02/13/2019', True, datetime(2019, 2, 13))])
def test_parse_delimited_date_swap_with_warning(all_parsers, date_string, dayfirst, expected):
    parser = all_parsers
    expected = DataFrame({0: [expected]}, dtype='datetime64[ns]')
    warning_msg = 'Parsing dates in .* format when dayfirst=.* was specified. Pass `dayfirst=.*` or specify a format to silence this warning.'
    result = parser.read_csv_check_warnings(UserWarning, warning_msg, StringIO(date_string), header=None, dayfirst=dayfirst, parse_dates=[0])
    tm.assert_frame_equal(result, expected)