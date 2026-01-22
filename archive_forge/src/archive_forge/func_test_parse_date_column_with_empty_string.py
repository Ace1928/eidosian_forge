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
def test_parse_date_column_with_empty_string(all_parsers):
    parser = all_parsers
    data = 'case,opdate\n7,10/18/2006\n7,10/18/2008\n621, '
    result = parser.read_csv(StringIO(data), parse_dates=['opdate'])
    expected_data = [[7, '10/18/2006'], [7, '10/18/2008'], [621, ' ']]
    expected = DataFrame(expected_data, columns=['case', 'opdate'])
    tm.assert_frame_equal(result, expected)