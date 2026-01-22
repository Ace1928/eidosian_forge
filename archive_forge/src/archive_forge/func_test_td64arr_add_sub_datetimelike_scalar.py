from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('cls', [Timestamp, datetime, np.datetime64])
def test_td64arr_add_sub_datetimelike_scalar(self, cls, box_with_array, tz_naive_fixture):
    tz = tz_naive_fixture
    dt_scalar = Timestamp('2012-01-01', tz=tz)
    if cls is datetime:
        ts = dt_scalar.to_pydatetime()
    elif cls is np.datetime64:
        if tz_naive_fixture is not None:
            pytest.skip(f'{cls} doesn support {tz_naive_fixture}')
        ts = dt_scalar.to_datetime64()
    else:
        ts = dt_scalar
    tdi = timedelta_range('1 day', periods=3)
    expected = pd.date_range('2012-01-02', periods=3, tz=tz)
    tdarr = tm.box_expected(tdi, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(ts + tdarr, expected)
    tm.assert_equal(tdarr + ts, expected)
    expected2 = pd.date_range('2011-12-31', periods=3, freq='-1D', tz=tz)
    expected2 = tm.box_expected(expected2, box_with_array)
    tm.assert_equal(ts - tdarr, expected2)
    tm.assert_equal(ts + -tdarr, expected2)
    msg = 'cannot subtract a datelike'
    with pytest.raises(TypeError, match=msg):
        tdarr - ts