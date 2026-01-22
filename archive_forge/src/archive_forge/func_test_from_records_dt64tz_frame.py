from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_dt64tz_frame(self):
    dti = date_range('2016-01-01', periods=10, tz='US/Pacific')
    df = DataFrame({i: dti for i in range(4)})
    with tm.assert_produces_warning(FutureWarning):
        res = DataFrame.from_records(df)
    tm.assert_frame_equal(res, df)