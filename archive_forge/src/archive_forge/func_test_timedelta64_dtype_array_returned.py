from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_timedelta64_dtype_array_returned(self):
    expected = np.array([31200, 45678, 10000], dtype='m8[ns]')
    td_index = to_timedelta([31200, 45678, 31200, 10000, 45678])
    result = algos.unique(td_index)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype
    s = Series(td_index)
    result = algos.unique(s)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype
    arr = s.values
    result = algos.unique(arr)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype