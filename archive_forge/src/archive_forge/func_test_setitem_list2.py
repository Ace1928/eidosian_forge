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
def test_setitem_list2(self):
    df = DataFrame(0, index=range(3), columns=['tt1', 'tt2'], dtype=int)
    df.loc[1, ['tt1', 'tt2']] = [1, 2]
    result = df.loc[df.index[1], ['tt1', 'tt2']]
    expected = Series([1, 2], df.columns, dtype=int, name=1)
    tm.assert_series_equal(result, expected)
    df['tt1'] = df['tt2'] = '0'
    df.loc[df.index[1], ['tt1', 'tt2']] = ['1', '2']
    result = df.loc[df.index[1], ['tt1', 'tt2']]
    expected = Series(['1', '2'], df.columns, name=1)
    tm.assert_series_equal(result, expected)