import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_incorrect_dtype_raises(self):
    depr_msg = 'DatetimeArray.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
            DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='category')
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
            DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='m8[s]')
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        with pytest.raises(ValueError, match="Unexpected value for 'dtype'."):
            DatetimeArray(np.array([1, 2, 3], dtype='i8'), dtype='M8[D]')