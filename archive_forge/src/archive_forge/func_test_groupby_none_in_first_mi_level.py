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
def test_groupby_none_in_first_mi_level():
    arr = [[None, 1, 0, 1], [2, 3, 2, 3]]
    ser = Series(1, index=MultiIndex.from_arrays(arr, names=['a', 'b']))
    result = ser.groupby(level=[0, 1]).sum()
    expected = Series([1, 2], MultiIndex.from_tuples([(0.0, 2), (1.0, 3)], names=['a', 'b']))
    tm.assert_series_equal(result, expected)