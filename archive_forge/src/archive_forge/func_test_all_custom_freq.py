from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('freq', [freq for freq in prefix_mapping if freq.startswith('C')])
def test_all_custom_freq(self, freq):
    bdate_range(START, END, freq=freq, weekmask='Mon Wed Fri', holidays=['2009-03-14'])
    bad_freq = freq + 'FOO'
    msg = f'invalid custom frequency string: {bad_freq}'
    with pytest.raises(ValueError, match=msg):
        bdate_range(START, END, freq=bad_freq)