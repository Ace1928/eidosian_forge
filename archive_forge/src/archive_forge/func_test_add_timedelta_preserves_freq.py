import calendar
from datetime import datetime
import locale
import unicodedata
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tseries.frequencies import to_offset
def test_add_timedelta_preserves_freq():
    tz = 'Canada/Eastern'
    dti = date_range(start=Timestamp('2019-03-26 00:00:00-0400', tz=tz), end=Timestamp('2020-10-17 00:00:00-0400', tz=tz), freq='D')
    result = dti + Timedelta(days=1)
    assert result.freq == dti.freq