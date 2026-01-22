import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('nano', [-1, 1000])
def test_timestamp_nano_range(nano):
    with pytest.raises(ValueError, match='nanosecond must be in 0..999'):
        Timestamp(year=2022, month=1, day=1, nanosecond=nano)