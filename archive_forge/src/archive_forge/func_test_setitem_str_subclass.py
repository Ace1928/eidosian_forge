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
def test_setitem_str_subclass(self):

    class mystring(str):
        pass
    data = ['2020-10-22 01:21:00+00:00']
    index = DatetimeIndex(data)
    df = DataFrame({'a': [1]}, index=index)
    df['b'] = 2
    df[mystring('c')] = 3
    expected = DataFrame({'a': [1], 'b': [2], mystring('c'): [3]}, index=index)
    tm.assert_equal(df, expected)