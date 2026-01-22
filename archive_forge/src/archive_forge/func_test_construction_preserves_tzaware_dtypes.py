import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
def test_construction_preserves_tzaware_dtypes(self, tz):
    dr = date_range('2011/1/1', '2012/1/1', freq='W-FRI')
    dr_tz = dr.tz_localize(tz)
    df = DataFrame({'A': 'foo', 'B': dr_tz}, index=dr)
    tz_expected = DatetimeTZDtype('ns', dr_tz.tzinfo)
    assert df['B'].dtype == tz_expected
    datetimes_naive = [ts.to_pydatetime() for ts in dr]
    datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
    df = DataFrame({'dr': dr})
    df['dr_tz'] = dr_tz
    df['datetimes_naive'] = datetimes_naive
    df['datetimes_with_tz'] = datetimes_with_tz
    result = df.dtypes
    expected = Series([np.dtype('datetime64[ns]'), DatetimeTZDtype(tz=tz), np.dtype('datetime64[ns]'), DatetimeTZDtype(tz=tz)], index=['dr', 'dr_tz', 'datetimes_naive', 'datetimes_with_tz'])
    tm.assert_series_equal(result, expected)