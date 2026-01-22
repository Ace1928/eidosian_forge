import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
def test_null_date(datapath):
    fname = datapath('io', 'sas', 'data', 'dates_null.sas7bdat')
    df = pd.read_sas(fname, encoding='utf-8')
    expected = pd.DataFrame({'datecol': np.array([datetime(9999, 12, 29), np.datetime64('NaT')], dtype='M8[s]'), 'datetimecol': np.array([datetime(9999, 12, 29, 23, 59, 59, 999000), np.datetime64('NaT')], dtype='M8[ms]')})
    if not IS64:
        expected.loc[0, 'datetimecol'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, expected)