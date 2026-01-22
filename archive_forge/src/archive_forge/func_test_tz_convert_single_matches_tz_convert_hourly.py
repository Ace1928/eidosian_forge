from datetime import datetime
import numpy as np
import pytest
from pytz import UTC
from pandas._libs.tslibs import (
from pandas import (
import pandas._testing as tm
def test_tz_convert_single_matches_tz_convert_hourly(tz_aware_fixture):
    tz = tz_aware_fixture
    tz_didx = date_range('2014-03-01', '2015-01-10', freq='h', tz=tz)
    naive_didx = date_range('2014-03-01', '2015-01-10', freq='h')
    _compare_utc_to_local(tz_didx)
    _compare_local_to_utc(tz_didx, naive_didx)