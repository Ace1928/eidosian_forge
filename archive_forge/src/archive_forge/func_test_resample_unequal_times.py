from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_resample_unequal_times(unit):
    start = datetime(1999, 3, 1, 5)
    end = datetime(2012, 7, 31, 4)
    bad_ind = date_range(start, end, freq='30min').as_unit(unit)
    df = DataFrame({'close': 1}, index=bad_ind)
    df.resample('YS').sum()