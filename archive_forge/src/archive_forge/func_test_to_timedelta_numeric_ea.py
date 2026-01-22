from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_numeric_ea(self, any_numeric_ea_dtype):
    ser = Series([1, pd.NA], dtype=any_numeric_ea_dtype)
    result = to_timedelta(ser)
    expected = Series([pd.Timedelta(1, unit='ns'), pd.NaT])
    tm.assert_series_equal(result, expected)