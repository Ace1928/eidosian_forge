from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_internals_not_updated_correctly(self):
    df = DataFrame({'bool_col': True, 'a': 1, 'b': 2.5}, index=MultiIndex.from_arrays([[1, 2], [1, 2]], names=['idx1', 'idx2']))
    idx = [(1, 1)]
    df['c'] = 3
    df.loc[idx, 'c'] = 0
    df.loc[idx, 'c']
    df.loc[idx, ['a', 'b']]
    df.loc[idx, 'c'] = 15
    result = df.loc[idx, 'c']
    expected = df = Series(15, index=MultiIndex.from_arrays([[1], [1]], names=['idx1', 'idx2']), name='c')
    tm.assert_series_equal(result, expected)