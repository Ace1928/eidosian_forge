from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
def test_round_subsecond(self):
    result = Timestamp('2016-10-17 12:00:00.0015').round('ms')
    expected = Timestamp('2016-10-17 12:00:00.002000')
    assert result == expected
    result = Timestamp('2016-10-17 12:00:00.00149').round('ms')
    expected = Timestamp('2016-10-17 12:00:00.001000')
    assert result == expected
    ts = Timestamp('2016-10-17 12:00:00.0015')
    for freq in ['us', 'ns']:
        assert ts == ts.round(freq)
    result = Timestamp('2016-10-17 12:00:00.001501031').round('10ns')
    expected = Timestamp('2016-10-17 12:00:00.001501030')
    assert result == expected