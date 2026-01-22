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
def test_datetime64_dtype_array_returned(self):
    expected = np.array(['2015-01-03T00:00:00.000000000', '2015-01-01T00:00:00.000000000'], dtype='M8[ns]')
    dt_index = to_datetime(['2015-01-03T00:00:00.000000000', '2015-01-01T00:00:00.000000000', '2015-01-01T00:00:00.000000000'])
    result = algos.unique(dt_index)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype
    s = Series(dt_index)
    result = algos.unique(s)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype
    arr = s.values
    result = algos.unique(arr)
    tm.assert_numpy_array_equal(result, expected)
    assert result.dtype == expected.dtype