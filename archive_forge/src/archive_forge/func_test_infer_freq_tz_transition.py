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
@pytest.mark.parametrize('date_pair', [['2013-11-02', '2013-11-5'], ['2014-03-08', '2014-03-11'], ['2014-01-01', '2014-01-03']])
@pytest.mark.parametrize('freq', ['h', '3h', '10min', '3601s', '3600001ms', '3600000001us', '3600000000001ns'])
def test_infer_freq_tz_transition(tz_naive_fixture, date_pair, freq):
    tz = tz_naive_fixture
    idx = date_range(date_pair[0], date_pair[1], freq=freq, tz=tz)
    assert idx.inferred_freq == freq