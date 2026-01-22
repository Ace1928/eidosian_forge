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
def test_groupby_multiindex_nat():
    values = [(pd.NaT, 'a'), (datetime(2012, 1, 2), 'a'), (datetime(2012, 1, 2), 'b'), (datetime(2012, 1, 3), 'a')]
    mi = MultiIndex.from_tuples(values, names=['date', None])
    ser = Series([3, 2, 2.5, 4], index=mi)
    result = ser.groupby(level=1).mean()
    expected = Series([3.0, 2.5], index=['a', 'b'])
    tm.assert_series_equal(result, expected)