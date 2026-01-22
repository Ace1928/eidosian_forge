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
def test_resample_axis1(unit):
    rng = date_range('1/1/2000', '2/29/2000').as_unit(unit)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, len(rng))), columns=rng, index=['a', 'b', 'c'])
    warning_msg = 'DataFrame.resample with axis=1 is deprecated.'
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        result = df.resample('ME', axis=1).mean()
    expected = df.T.resample('ME').mean().T
    tm.assert_frame_equal(result, expected)