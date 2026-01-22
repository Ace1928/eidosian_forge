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
@pytest.mark.parametrize('func', ['sum', 'cumsum', 'cumprod', 'prod'])
def test_groupby_avoid_casting_to_float(func):
    val = 922337203685477580
    df = DataFrame({'a': 1, 'b': [val]})
    result = getattr(df.groupby('a'), func)() - val
    expected = DataFrame({'b': [0]}, index=Index([1], name='a'))
    if func in ['cumsum', 'cumprod']:
        expected = expected.reset_index(drop=True)
    tm.assert_frame_equal(result, expected)