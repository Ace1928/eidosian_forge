import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
def test_is_unitless():
    dtype = np.dtype('M8[ns]')
    assert not is_unitless(dtype)
    dtype = np.dtype('datetime64')
    assert is_unitless(dtype)
    dtype = np.dtype('m8[ns]')
    assert not is_unitless(dtype)
    dtype = np.dtype('timedelta64')
    assert is_unitless(dtype)
    msg = 'dtype must be datetime64 or timedelta64'
    with pytest.raises(ValueError, match=msg):
        is_unitless(np.dtype(np.int64))
    msg = "Argument 'dtype' has incorrect type"
    with pytest.raises(TypeError, match=msg):
        is_unitless('foo')