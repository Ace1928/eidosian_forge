import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.np_datetime import (
import pandas._testing as tm
def test_pass_non_dt64_array(self):
    arr = np.arange(5)
    dtype = np.dtype('M8[ns]')
    msg = 'astype_overflowsafe values.dtype and dtype must be either both-datetime64 or both-timedelta64'
    with pytest.raises(TypeError, match=msg):
        astype_overflowsafe(arr, dtype, copy=True)
    with pytest.raises(TypeError, match=msg):
        astype_overflowsafe(arr, dtype, copy=False)