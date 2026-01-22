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
def test_non_nano_dt64_addsub_np_nat_scalars_unitless():
    ser = Series([1233242342344, 232432434324, 332434242344], dtype='datetime64[ms]')
    result = ser - np.datetime64('nat')
    expected = Series([NaT] * 3, dtype='timedelta64[ns]')
    tm.assert_series_equal(result, expected)
    result = ser + np.timedelta64('nat')
    expected = Series([NaT] * 3, dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)