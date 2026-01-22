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
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
def test_period_deprecated_freq(self):
    cases = {'M': ['MTH', 'MONTH', 'MONTHLY', 'Mth', 'month', 'monthly'], 'B': ['BUS', 'BUSINESS', 'BUSINESSLY', 'WEEKDAY', 'bus'], 'D': ['DAY', 'DLY', 'DAILY', 'Day', 'Dly', 'Daily'], 'h': ['HR', 'HOUR', 'HRLY', 'HOURLY', 'hr', 'Hour', 'HRly'], 'min': ['minute', 'MINUTE', 'MINUTELY', 'minutely'], 's': ['sec', 'SEC', 'SECOND', 'SECONDLY', 'second'], 'ms': ['MILLISECOND', 'MILLISECONDLY', 'millisecond'], 'us': ['MICROSECOND', 'MICROSECONDLY', 'microsecond'], 'ns': ['NANOSECOND', 'NANOSECONDLY', 'nanosecond']}
    msg = INVALID_FREQ_ERR_MSG
    for exp, freqs in cases.items():
        for freq in freqs:
            with pytest.raises(ValueError, match=msg):
                Period('2016-03-01 09:00', freq=freq)
            with pytest.raises(ValueError, match=msg):
                Period(ordinal=1, freq=freq)
        p1 = Period('2016-03-01 09:00', freq=exp)
        p2 = Period(ordinal=1, freq=exp)
        assert isinstance(p1, Period)
        assert isinstance(p2, Period)