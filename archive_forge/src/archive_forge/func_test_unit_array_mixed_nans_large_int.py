import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
def test_unit_array_mixed_nans_large_int(self, cache):
    values = [1420043460000000000000000, iNaT, NaT, np.nan, 'NaT']
    result = to_datetime(values, errors='ignore', unit='s', cache=cache)
    expected = Index([1420043460000000000000000, NaT, NaT, NaT, NaT], dtype=object)
    tm.assert_index_equal(result, expected)
    result = to_datetime(values, errors='coerce', unit='s', cache=cache)
    expected = DatetimeIndex(['NaT', 'NaT', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    msg = "cannot convert input 1420043460000000000000000 with the unit 's'"
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        to_datetime(values, errors='raise', unit='s', cache=cache)