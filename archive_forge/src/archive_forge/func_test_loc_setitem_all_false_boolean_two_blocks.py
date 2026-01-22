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
def test_loc_setitem_all_false_boolean_two_blocks(self):
    df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': 'a'})
    expected = df.copy()
    indexer = Series([False, False], name='c')
    df.loc[indexer, ['b']] = DataFrame({'b': [5, 6]}, index=[0, 1])
    tm.assert_frame_equal(df, expected)