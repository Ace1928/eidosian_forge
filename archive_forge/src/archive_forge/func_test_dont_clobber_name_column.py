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
def test_dont_clobber_name_column():
    df = DataFrame({'key': ['a', 'a', 'a', 'b', 'b', 'b'], 'name': ['foo', 'bar', 'baz'] * 2})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('key', group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df)