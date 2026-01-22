from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_as_index_select_column():
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=['A', 'B'])
    result = df.groupby('A', as_index=False)['B'].get_group(1)
    expected = Series([2, 4], name='B')
    tm.assert_series_equal(result, expected)
    result = df.groupby('A', as_index=False, group_keys=True)['B'].apply(lambda x: x.cumsum())
    expected = Series([2, 6, 6], name='B', index=MultiIndex.from_tuples([(0, 0), (0, 1), (1, 2)]))
    tm.assert_series_equal(result, expected)