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
def test_tuple_as_grouping():
    df = DataFrame({('a', 'b'): [1, 1, 1, 1], 'a': [2, 2, 2, 2], 'b': [2, 2, 2, 2], 'c': [1, 1, 1, 1]})
    with pytest.raises(KeyError, match="('a', 'b')"):
        df[['a', 'b', 'c']].groupby(('a', 'b'))
    result = df.groupby(('a', 'b'))['c'].sum()
    expected = Series([4], name='c', index=Index([1], name=('a', 'b')))
    tm.assert_series_equal(result, expected)