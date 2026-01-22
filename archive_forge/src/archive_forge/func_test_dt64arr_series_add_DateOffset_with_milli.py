from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_dt64arr_series_add_DateOffset_with_milli(self):
    dti = DatetimeIndex(['2000-01-01 00:00:00.012345678', '2000-01-31 00:00:00.012345678', '2000-02-29 00:00:00.012345678'], dtype='datetime64[ns]')
    result = dti + DateOffset(milliseconds=4)
    expected = DatetimeIndex(['2000-01-01 00:00:00.016345678', '2000-01-31 00:00:00.016345678', '2000-02-29 00:00:00.016345678'], dtype='datetime64[ns]')
    tm.assert_index_equal(result, expected)
    result = dti + DateOffset(days=1, milliseconds=4)
    expected = DatetimeIndex(['2000-01-02 00:00:00.016345678', '2000-02-01 00:00:00.016345678', '2000-03-01 00:00:00.016345678'], dtype='datetime64[ns]')
    tm.assert_index_equal(result, expected)