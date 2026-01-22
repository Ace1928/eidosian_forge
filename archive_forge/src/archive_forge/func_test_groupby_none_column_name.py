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
def test_groupby_none_column_name():
    df = DataFrame({None: [1, 1, 2, 2], 'b': [1, 1, 2, 3], 'c': [4, 5, 6, 7]})
    result = df.groupby(by=[None]).sum()
    expected = DataFrame({'b': [2, 5], 'c': [9, 13]}, index=Index([1, 2], name=None))
    tm.assert_frame_equal(result, expected)