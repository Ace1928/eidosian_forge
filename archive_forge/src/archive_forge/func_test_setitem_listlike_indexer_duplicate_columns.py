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
def test_setitem_listlike_indexer_duplicate_columns(self):
    df = DataFrame([[1, 2, 3]], columns=['a', 'b', 'b'])
    rhs = DataFrame([[10, 11, 12]], columns=['a', 'b', 'b'])
    df[['a', 'b']] = rhs
    expected = DataFrame([[10, 11, 12]], columns=['a', 'b', 'b'])
    tm.assert_frame_equal(df, expected)
    df[['c', 'b']] = rhs
    expected = DataFrame([[10, 11, 12, 10]], columns=['a', 'b', 'b', 'c'])
    tm.assert_frame_equal(df, expected)