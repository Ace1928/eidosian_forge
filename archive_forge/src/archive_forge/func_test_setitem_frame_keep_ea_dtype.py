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
def test_setitem_frame_keep_ea_dtype(self, any_numeric_ea_dtype):
    df = DataFrame(columns=['a', 'b'], data=[[1, 2], [3, 4]])
    df['c'] = DataFrame({'a': [10, 11]}, dtype=any_numeric_ea_dtype)
    expected = DataFrame({'a': [1, 3], 'b': [2, 4], 'c': Series([10, 11], dtype=any_numeric_ea_dtype)})
    tm.assert_frame_equal(df, expected)