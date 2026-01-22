import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_mismatched_timezone_raises(self):
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        arr = DatetimeArray(np.array(['2000-01-01T06:00:00'], dtype='M8[ns]'), dtype=DatetimeTZDtype(tz='US/Central'))
    dtype = DatetimeTZDtype(tz='US/Eastern')
    msg = 'dtype=datetime64\\[ns.*\\] does not match data dtype datetime64\\[ns.*\\]'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr, dtype=dtype)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr, dtype=np.dtype('M8[ns]'))
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(TypeError, match=msg):
            DatetimeArray(arr.tz_localize(None), dtype=arr.dtype)