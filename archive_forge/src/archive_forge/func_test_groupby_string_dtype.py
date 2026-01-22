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
def test_groupby_string_dtype():
    df = DataFrame({'str_col': ['a', 'b', 'c', 'a'], 'num_col': [1, 2, 3, 2]})
    df['str_col'] = df['str_col'].astype('string')
    expected = DataFrame({'str_col': ['a', 'b', 'c'], 'num_col': [1.5, 2.0, 3.0]})
    expected['str_col'] = expected['str_col'].astype('string')
    grouped = df.groupby('str_col', as_index=False)
    result = grouped.mean()
    tm.assert_frame_equal(result, expected)