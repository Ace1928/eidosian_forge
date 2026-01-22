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
def test_groupby_ffill_with_duplicated_index():
    df = DataFrame({'a': [1, 2, 3, 4, np.nan, np.nan]}, index=[0, 1, 2, 0, 1, 2])
    result = df.groupby(level=0).ffill()
    expected = DataFrame({'a': [1, 2, 3, 4, 2, 3]}, index=[0, 1, 2, 0, 1, 2])
    tm.assert_frame_equal(result, expected, check_dtype=False)