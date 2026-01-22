from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ts, expected', [('1970-01-01 00:00:00', 0), ('1970-01-01 00:00:00.000001', 1), ('1970-01-01 00:00:00.00001', 10), ('1970-01-01 00:00:00.499', 499000), ('1999-12-31 23:59:59.999', 999000), ('1999-12-31 23:59:59.999999', 999999), ('2050-12-31 23:59:59.5', 500000), ('2050-12-31 23:59:59.500001', 500001), ('2050-12-31 23:59:59.123456', 123456)])
@pytest.mark.parametrize('freq', [None, 'us', 'ns'])
def test_to_timestamp_microsecond(self, ts, expected, freq):
    result = Period(ts).to_timestamp(freq=freq).microsecond
    assert result == expected