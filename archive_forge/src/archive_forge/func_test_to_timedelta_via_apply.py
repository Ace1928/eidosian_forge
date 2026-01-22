from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_via_apply(self):
    expected = Series([np.timedelta64(1, 's')])
    result = Series(['00:00:01']).apply(to_timedelta)
    tm.assert_series_equal(result, expected)
    result = Series([to_timedelta('00:00:01')])
    tm.assert_series_equal(result, expected)