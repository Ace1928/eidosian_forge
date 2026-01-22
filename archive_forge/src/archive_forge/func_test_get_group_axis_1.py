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
def test_get_group_axis_1():
    df = DataFrame({'col1': [0, 3, 2, 3], 'col2': [4, 1, 6, 7], 'col3': [3, 8, 2, 10], 'col4': [1, 13, 6, 15], 'col5': [-4, 5, 6, -7]})
    with tm.assert_produces_warning(FutureWarning, match='deprecated'):
        grouped = df.groupby(axis=1, by=[1, 2, 3, 2, 1])
    result = grouped.get_group(1)
    expected = DataFrame({'col1': [0, 3, 2, 3], 'col5': [-4, 5, 6, -7]})
    tm.assert_frame_equal(result, expected)