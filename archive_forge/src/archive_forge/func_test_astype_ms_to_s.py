from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_astype_ms_to_s(self, index_or_series):
    scalar = Timedelta(days=31)
    td = index_or_series([scalar, scalar, scalar + timedelta(minutes=5, seconds=3), NaT], dtype='m8[ns]')
    exp_values = np.asarray(td).astype('m8[s]')
    exp_tda = TimedeltaArray._simple_new(exp_values, dtype=exp_values.dtype)
    expected = index_or_series(exp_tda)
    assert expected.dtype == 'm8[s]'
    result = td.astype('timedelta64[s]')
    tm.assert_equal(result, expected)