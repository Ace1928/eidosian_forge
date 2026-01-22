import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_date_format(self, datetime_frame):
    with tm.ensure_clean('__tmp_to_csv_date_format__') as path:
        dt_index = datetime_frame.index
        datetime_frame = DataFrame({'A': dt_index, 'B': dt_index.shift(1)}, index=dt_index)
        datetime_frame.to_csv(path, date_format='%Y%m%d')
        test = read_csv(path, index_col=0)
        datetime_frame_int = datetime_frame.map(lambda x: int(x.strftime('%Y%m%d')))
        datetime_frame_int.index = datetime_frame_int.index.map(lambda x: int(x.strftime('%Y%m%d')))
        tm.assert_frame_equal(test, datetime_frame_int)
        datetime_frame.to_csv(path, date_format='%Y-%m-%d')
        test = read_csv(path, index_col=0)
        datetime_frame_str = datetime_frame.map(lambda x: x.strftime('%Y-%m-%d'))
        datetime_frame_str.index = datetime_frame_str.index.map(lambda x: x.strftime('%Y-%m-%d'))
        tm.assert_frame_equal(test, datetime_frame_str)
        datetime_frame_columns = datetime_frame.T
        datetime_frame_columns.to_csv(path, date_format='%Y%m%d')
        test = read_csv(path, index_col=0)
        datetime_frame_columns = datetime_frame_columns.map(lambda x: int(x.strftime('%Y%m%d')))
        datetime_frame_columns.columns = datetime_frame_columns.columns.map(lambda x: x.strftime('%Y%m%d'))
        tm.assert_frame_equal(test, datetime_frame_columns)
        nat_index = to_datetime(['NaT'] * 10 + ['2000-01-01', '2000-01-01', '2000-01-01'])
        nat_frame = DataFrame({'A': nat_index}, index=nat_index)
        nat_frame.to_csv(path, date_format='%Y-%m-%d')
        test = read_csv(path, parse_dates=[0, 1], index_col=0)
        tm.assert_frame_equal(test, nat_frame)