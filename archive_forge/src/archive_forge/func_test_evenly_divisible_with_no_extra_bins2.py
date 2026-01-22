from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_evenly_divisible_with_no_extra_bins2(self):
    index = date_range(start='2001-5-4', periods=28)
    df = DataFrame([{'REST_KEY': 1, 'DLY_TRN_QT': 80, 'DLY_SLS_AMT': 90, 'COOP_DLY_TRN_QT': 30, 'COOP_DLY_SLS_AMT': 20}] * 28 + [{'REST_KEY': 2, 'DLY_TRN_QT': 70, 'DLY_SLS_AMT': 10, 'COOP_DLY_TRN_QT': 50, 'COOP_DLY_SLS_AMT': 20}] * 28, index=index.append(index)).sort_index()
    index = date_range('2001-5-4', periods=4, freq='7D')
    expected = DataFrame([{'REST_KEY': 14, 'DLY_TRN_QT': 14, 'DLY_SLS_AMT': 14, 'COOP_DLY_TRN_QT': 14, 'COOP_DLY_SLS_AMT': 14}] * 4, index=index)
    result = df.resample('7D').count()
    tm.assert_frame_equal(result, expected)
    expected = DataFrame([{'REST_KEY': 21, 'DLY_TRN_QT': 1050, 'DLY_SLS_AMT': 700, 'COOP_DLY_TRN_QT': 560, 'COOP_DLY_SLS_AMT': 280}] * 4, index=index)
    result = df.resample('7D').sum()
    tm.assert_frame_equal(result, expected)