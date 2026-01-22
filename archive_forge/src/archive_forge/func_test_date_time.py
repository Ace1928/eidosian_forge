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
def test_date_time(datapath):
    fname = datapath('io', 'sas', 'data', 'datetime.sas7bdat')
    df = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'datetime.csv')
    df0 = pd.read_csv(fname, parse_dates=['Date1', 'Date2', 'DateTime', 'DateTimeHi', 'Taiw'])
    df[df.columns[3]] = df.iloc[:, 3].dt.round('us')
    df0['Date1'] = df0['Date1'].astype('M8[s]')
    df0['Date2'] = df0['Date2'].astype('M8[s]')
    df0['DateTime'] = df0['DateTime'].astype('M8[ms]')
    df0['Taiw'] = df0['Taiw'].astype('M8[s]')
    res = df0['DateTimeHi'].astype('M8[us]').dt.round('ms')
    df0['DateTimeHi'] = res.astype('M8[ms]')
    if not IS64:
        df0.loc[0, 'DateTimeHi'] += np.timedelta64(1, 'ms')
        df0.loc[[2, 3], 'DateTimeHi'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, df0)