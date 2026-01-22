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
def test_setitem_frame_duplicate_columns(self):
    cols = ['A', 'B', 'C'] * 2
    df = DataFrame(index=range(3), columns=cols)
    df.loc[0, 'A'] = (0, 3)
    df.loc[:, 'B'] = (1, 4)
    df['C'] = (2, 5)
    expected = DataFrame([[0, 1, 2, 3, 4, 5], [np.nan, 1, 2, np.nan, 4, 5], [np.nan, 1, 2, np.nan, 4, 5]], dtype='object')
    expected[2] = expected[2].astype(np.int64)
    expected[5] = expected[5].astype(np.int64)
    expected.columns = cols
    tm.assert_frame_equal(df, expected)