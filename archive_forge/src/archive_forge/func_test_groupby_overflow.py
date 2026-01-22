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
@pytest.mark.parametrize('val, dtype', [(111, 'int'), (222, 'uint')])
def test_groupby_overflow(val, dtype):
    df = DataFrame({'a': 1, 'b': [val, val]}, dtype=f'{dtype}8')
    result = df.groupby('a').sum()
    expected = DataFrame({'b': [val * 2]}, index=Index([1], name='a', dtype=f'{dtype}8'), dtype=f'{dtype}64')
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a').cumsum()
    expected = DataFrame({'b': [val, val * 2]}, dtype=f'{dtype}64')
    tm.assert_frame_equal(result, expected)
    result = df.groupby('a').prod()
    expected = DataFrame({'b': [val * val]}, index=Index([1], name='a', dtype=f'{dtype}8'), dtype=f'{dtype}64')
    tm.assert_frame_equal(result, expected)