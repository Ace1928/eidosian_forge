import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_bool_dtype_raises(self):
    arr = np.array([1, 2, 3], dtype='bool')
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    msg = "Unexpected value for 'dtype': 'bool'. Must be"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match=msg):
            DatetimeArray(arr)
    msg = 'dtype bool cannot be converted to datetime64\\[ns\\]'
    with pytest.raises(TypeError, match=msg):
        DatetimeArray._from_sequence(arr, dtype='M8[ns]')
    with pytest.raises(TypeError, match=msg):
        pd.DatetimeIndex(arr)
    with pytest.raises(TypeError, match=msg):
        pd.to_datetime(arr)