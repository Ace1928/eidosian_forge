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
@pytest.mark.parametrize('freq, freq_depr, freq_res, freq_depr_res, data', [('2Q', '2q', '2Y', '2y', [0.5]), ('2M', '2m', '2Q', '2q', [1.0, 3.0])])
def test_resample_lowercase_frequency_deprecated(self, freq, freq_depr, freq_res, freq_depr_res, data):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version. Please use '{freq[1:]}' instead."
    depr_msg_res = f"'{freq_depr_res[1:]}' is deprecated and will be removed in a "
    f"future version. Please use '{freq_res[1:]}' instead."
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        rng_l = period_range('2020-01-01', '2020-08-01', freq=freq_depr)
    ser = Series(np.arange(len(rng_l)), index=rng_l)
    rng = period_range('2020-01-01', '2020-08-01', freq=freq_res)
    expected = Series(data=data, index=rng)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg_res):
        result = ser.resample(freq_depr_res).mean()
    tm.assert_series_equal(result, expected)