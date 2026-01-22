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
@pytest.mark.parametrize('f', ['sum', 'mean', 'prod', 'min', 'max', 'var'])
def test_resample_frame_basic_cy_funcs(f, unit):
    df = DataFrame(np.random.default_rng(2).standard_normal((50, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=50, freq='B'))
    df.index = df.index.as_unit(unit)
    b = Grouper(freq='ME')
    g = df.groupby(b)
    g._cython_agg_general(f, alt=None, numeric_only=True)