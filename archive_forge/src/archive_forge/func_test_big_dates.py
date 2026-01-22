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
def test_big_dates(self, datapath):
    yr = [1960, 2000, 9999, 100, 2262, 1677]
    mo = [1, 1, 12, 1, 4, 9]
    dd = [1, 1, 31, 1, 22, 23]
    hr = [0, 0, 23, 0, 0, 0]
    mm = [0, 0, 59, 0, 0, 0]
    ss = [0, 0, 59, 0, 0, 0]
    expected = []
    for year, month, day, hour, minute, second in zip(yr, mo, dd, hr, mm, ss):
        row = []
        for j in range(7):
            if j == 0:
                row.append(datetime(year, month, day, hour, minute, second))
            elif j == 6:
                row.append(datetime(year, 1, 1))
            else:
                row.append(datetime(year, month, day))
        expected.append(row)
    expected.append([pd.NaT] * 7)
    columns = ['date_tc', 'date_td', 'date_tw', 'date_tm', 'date_tq', 'date_th', 'date_ty']
    expected[2][2] = datetime(9999, 12, 24)
    expected[2][3] = datetime(9999, 12, 1)
    expected[2][4] = datetime(9999, 10, 1)
    expected[2][5] = datetime(9999, 7, 1)
    expected[4][2] = datetime(2262, 4, 16)
    expected[4][3] = expected[4][4] = datetime(2262, 4, 1)
    expected[4][5] = expected[4][6] = datetime(2262, 1, 1)
    expected[5][2] = expected[5][3] = expected[5][4] = datetime(1677, 10, 1)
    expected[5][5] = expected[5][6] = datetime(1678, 1, 1)
    expected = DataFrame(expected, columns=columns, dtype=object)
    parsed_115 = read_stata(datapath('io', 'data', 'stata', 'stata9_115.dta'))
    parsed_117 = read_stata(datapath('io', 'data', 'stata', 'stata9_117.dta'))
    tm.assert_frame_equal(expected, parsed_115, check_datetimelike_compat=True)
    tm.assert_frame_equal(expected, parsed_117, check_datetimelike_compat=True)
    date_conversion = {c: c[-2:] for c in columns}
    with tm.ensure_clean() as path:
        expected.index.name = 'index'
        expected.to_stata(path, convert_dates=date_conversion)
        written_and_read_again = self.read_dta(path)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected.set_index(expected.index.astype(np.int32)), check_datetimelike_compat=True)