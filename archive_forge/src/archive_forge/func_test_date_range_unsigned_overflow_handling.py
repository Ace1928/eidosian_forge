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
def test_date_range_unsigned_overflow_handling(self):
    dti = date_range(start='1677-09-22', end='2262-04-11', freq='D')
    dti2 = date_range(start=dti[0], periods=len(dti), freq='D')
    assert dti2.equals(dti)
    dti3 = date_range(end=dti[-1], periods=len(dti), freq='D')
    assert dti3.equals(dti)