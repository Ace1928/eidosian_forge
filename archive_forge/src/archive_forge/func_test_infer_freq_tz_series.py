from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
def test_infer_freq_tz_series(tz_naive_fixture):
    tz = tz_naive_fixture
    idx = date_range('2021-01-01', '2021-01-04', tz=tz)
    series = idx.to_series().reset_index(drop=True)
    inferred_freq = frequencies.infer_freq(series)
    assert inferred_freq == 'D'