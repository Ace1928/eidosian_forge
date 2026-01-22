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
@pytest.mark.parametrize('freq, freq_depr', [('2BME', '2BM'), ('2BQE', '2BQ'), ('2BQE-MAR', '2BQ-MAR')])
def test_resample_BM_BQ_deprecated(freq, freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    s = Series(range(10), index=date_range('20130101', freq='d', periods=10))
    expected = s.resample(freq).mean()
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = s.resample(freq_depr).mean()
    tm.assert_series_equal(result, expected)