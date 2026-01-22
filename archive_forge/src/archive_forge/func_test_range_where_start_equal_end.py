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
def test_range_where_start_equal_end(self, inclusive_endpoints_fixture):
    start = '2021-09-02'
    end = '2021-09-02'
    result = date_range(start=start, end=end, freq='D', inclusive=inclusive_endpoints_fixture)
    both_range = date_range(start=start, end=end, freq='D', inclusive='both')
    if inclusive_endpoints_fixture == 'neither':
        expected = both_range[1:-1]
    elif inclusive_endpoints_fixture in ('left', 'right', 'both'):
        expected = both_range[:]
    tm.assert_index_equal(result, expected)