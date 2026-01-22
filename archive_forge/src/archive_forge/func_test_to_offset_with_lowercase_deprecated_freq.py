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
def test_to_offset_with_lowercase_deprecated_freq(self) -> None:
    msg = "'m' is deprecated and will be removed in a future version, please use 'ME' instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = date_range('2010-01-01', periods=2, freq='m')
    expected = DatetimeIndex(['2010-01-31', '2010-02-28'], freq='ME')
    tm.assert_index_equal(result, expected)