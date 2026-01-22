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
@pytest.mark.parametrize('day', DAYS)
@pytest.mark.parametrize('num', range(10, 17))
def test_period_cons_weekly(self, num, day):
    daystr = f'2011-02-{num}'
    freq = f'W-{day}'
    result = Period(daystr, freq=freq)
    expected = Period(daystr, freq='D').asfreq(freq)
    assert result == expected
    assert isinstance(result, Period)