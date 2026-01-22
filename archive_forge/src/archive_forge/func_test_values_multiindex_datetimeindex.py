import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_values_multiindex_datetimeindex():
    ints = np.arange(10 ** 18, 10 ** 18 + 5)
    naive = pd.DatetimeIndex(ints)
    aware = pd.DatetimeIndex(ints, tz='US/Central')
    idx = MultiIndex.from_arrays([naive, aware])
    result = idx.values
    outer = pd.DatetimeIndex([x[0] for x in result])
    tm.assert_index_equal(outer, naive)
    inner = pd.DatetimeIndex([x[1] for x in result])
    tm.assert_index_equal(inner, aware)
    result = idx[:2].values
    outer = pd.DatetimeIndex([x[0] for x in result])
    tm.assert_index_equal(outer, naive[:2])
    inner = pd.DatetimeIndex([x[1] for x in result])
    tm.assert_index_equal(inner, aware[:2])