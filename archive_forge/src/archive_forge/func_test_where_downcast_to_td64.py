from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_downcast_to_td64():
    ser = Series([1, 2, 3])
    mask = np.array([False, False, False])
    td = pd.Timedelta(days=1)
    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.where(mask, td)
    expected = Series([td, td, td], dtype='m8[ns]')
    tm.assert_series_equal(res, expected)
    with pd.option_context('future.no_silent_downcasting', True):
        with tm.assert_produces_warning(None, match=msg):
            res2 = ser.where(mask, td)
    expected2 = expected.astype(object)
    tm.assert_series_equal(res2, expected2)