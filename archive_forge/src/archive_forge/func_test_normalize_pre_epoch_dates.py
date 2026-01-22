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
def test_normalize_pre_epoch_dates(self):
    result = Timestamp('1969-01-01 09:00:00').normalize()
    expected = Timestamp('1969-01-01 00:00:00')
    assert result == expected