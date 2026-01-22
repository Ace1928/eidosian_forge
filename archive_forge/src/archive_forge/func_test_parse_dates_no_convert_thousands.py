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
@pytest.mark.parametrize('data,kwargs,expected', [('a\n04.15.2016', {'parse_dates': ['a']}, DataFrame([datetime(2016, 4, 15)], columns=['a'])), ('a\n04.15.2016', {'parse_dates': True, 'index_col': 0}, DataFrame(index=DatetimeIndex(['2016-04-15'], name='a'), columns=[])), ('a,b\n04.15.2016,09.16.2013', {'parse_dates': ['a', 'b']}, DataFrame([[datetime(2016, 4, 15), datetime(2013, 9, 16)]], columns=['a', 'b'])), ('a,b\n04.15.2016,09.16.2013', {'parse_dates': True, 'index_col': [0, 1]}, DataFrame(index=MultiIndex.from_tuples([(datetime(2016, 4, 15), datetime(2013, 9, 16))], names=['a', 'b']), columns=[]))])
def test_parse_dates_no_convert_thousands(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), thousands='.', **kwargs)
    tm.assert_frame_equal(result, expected)