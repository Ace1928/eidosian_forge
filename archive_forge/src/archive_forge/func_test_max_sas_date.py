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
def test_max_sas_date(datapath):
    fname = datapath('io', 'sas', 'data', 'max_sas_date.sas7bdat')
    df = pd.read_sas(fname, encoding='iso-8859-1')
    expected = pd.DataFrame({'text': ['max', 'normal'], 'dt_as_float': [253717747199.999, 1880323199.999], 'dt_as_dt': np.array([datetime(9999, 12, 29, 23, 59, 59, 999000), datetime(2019, 8, 1, 23, 59, 59, 999000)], dtype='M8[ms]'), 'date_as_float': [2936547.0, 21762.0], 'date_as_date': np.array([datetime(9999, 12, 29), datetime(2019, 8, 1)], dtype='M8[s]')}, columns=['text', 'dt_as_float', 'dt_as_dt', 'date_as_float', 'date_as_date'])
    if not IS64:
        expected.loc[:, 'dt_as_dt'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, expected)