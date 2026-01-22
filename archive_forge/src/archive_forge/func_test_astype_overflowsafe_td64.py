import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
def test_astype_overflowsafe_td64(self):
    dtype = np.dtype('m8[ns]')
    dt = np.datetime64('2262-04-05', 'D')
    arr = dt + np.arange(10, dtype='m8[D]')
    arr = arr.view('m8[D]')
    wrong = arr.astype(dtype)
    roundtrip = wrong.astype(arr.dtype)
    assert not (wrong == roundtrip).all()
    msg = 'Cannot convert 106752 days to timedelta64\\[ns\\] without overflow'
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        astype_overflowsafe(arr, dtype)
    dtype2 = np.dtype('m8[us]')
    result = astype_overflowsafe(arr, dtype2)
    expected = arr.astype(dtype2)
    tm.assert_numpy_array_equal(result, expected)