from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_loc_setitem_ea_dtype(self):
    df = DataFrame({'a': np.array([10], dtype='i8')})
    df.loc[:, 'a'] = Series([11], dtype='Int64')
    expected = DataFrame({'a': np.array([11], dtype='i8')})
    tm.assert_frame_equal(df, expected)
    df = DataFrame({'a': np.array([10], dtype='i8')})
    df.iloc[:, 0] = Series([11], dtype='Int64')
    tm.assert_frame_equal(df, expected)