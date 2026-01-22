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
def test_groupby_as_index_select_column_sum_empty_df():
    df = DataFrame(columns=Index(['A', 'B', 'C'], name='alpha'))
    left = df.groupby(by='A', as_index=False)['B'].sum(numeric_only=False)
    expected = DataFrame(columns=df.columns[:2], index=range(0))
    expected.columns.names = [None]
    tm.assert_frame_equal(left, expected)