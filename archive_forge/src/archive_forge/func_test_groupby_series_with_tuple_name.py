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
def test_groupby_series_with_tuple_name():
    ser = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=('a', 'a'))
    ser.index.name = ('b', 'b')
    result = ser.groupby(level=0).last()
    expected = Series([2, 4], index=[1, 2], name=('a', 'a'))
    expected.index.name = ('b', 'b')
    tm.assert_series_equal(result, expected)