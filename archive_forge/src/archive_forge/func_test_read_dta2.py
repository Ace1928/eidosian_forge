import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
def test_read_dta2(self, datapath):
    expected = DataFrame.from_records([(datetime(2006, 11, 19, 23, 13, 20), 1479596223000, datetime(2010, 1, 20), datetime(2010, 1, 8), datetime(2010, 1, 1), datetime(1974, 7, 1), datetime(2010, 1, 1), datetime(2010, 1, 1)), (datetime(1959, 12, 31, 20, 3, 20), -1479590, datetime(1953, 10, 2), datetime(1948, 6, 10), datetime(1955, 1, 1), datetime(1955, 7, 1), datetime(1955, 1, 1), datetime(2, 1, 1)), (pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT)], columns=['datetime_c', 'datetime_big_c', 'date', 'weekly_date', 'monthly_date', 'quarterly_date', 'half_yearly_date', 'yearly_date'])
    expected['yearly_date'] = expected['yearly_date'].astype('O')
    path1 = datapath('io', 'data', 'stata', 'stata2_114.dta')
    path2 = datapath('io', 'data', 'stata', 'stata2_115.dta')
    path3 = datapath('io', 'data', 'stata', 'stata2_117.dta')
    with tm.assert_produces_warning(UserWarning):
        parsed_114 = self.read_dta(path1)
    with tm.assert_produces_warning(UserWarning):
        parsed_115 = self.read_dta(path2)
    with tm.assert_produces_warning(UserWarning):
        parsed_117 = self.read_dta(path3)
    tm.assert_frame_equal(parsed_114, expected, check_datetimelike_compat=True)
    tm.assert_frame_equal(parsed_115, expected, check_datetimelike_compat=True)
    tm.assert_frame_equal(parsed_117, expected, check_datetimelike_compat=True)