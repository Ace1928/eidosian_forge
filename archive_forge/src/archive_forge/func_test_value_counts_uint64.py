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
def test_value_counts_uint64(self):
    arr = np.array([2 ** 63], dtype=np.uint64)
    expected = Series([1], index=[2 ** 63], name='count')
    msg = 'pandas.value_counts is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = algos.value_counts(arr)
    tm.assert_series_equal(result, expected)
    arr = np.array([-1, 2 ** 63], dtype=object)
    expected = Series([1, 1], index=[-1, 2 ** 63], name='count')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = algos.value_counts(arr)
    tm.assert_series_equal(result, expected)