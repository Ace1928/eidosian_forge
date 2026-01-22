from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq', ['D', 'YE'])
def test_tz_convert_single_matches_tz_convert(tz_aware_fixture, freq):
    tz = tz_aware_fixture
    tz_didx = date_range('2018-01-01', '2020-01-01', freq=freq, tz=tz)
    naive_didx = date_range('2018-01-01', '2020-01-01', freq=freq)
    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)