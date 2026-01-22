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
def test_groupby_filtered_df_std():
    dicts = [{'filter_col': False, 'groupby_col': True, 'bool_col': True, 'float_col': 10.5}, {'filter_col': True, 'groupby_col': True, 'bool_col': True, 'float_col': 20.5}, {'filter_col': True, 'groupby_col': True, 'bool_col': True, 'float_col': 30.5}]
    df = DataFrame(dicts)
    df_filter = df[df['filter_col'] == True]
    dfgb = df_filter.groupby('groupby_col')
    result = dfgb.std()
    expected = DataFrame([[0.0, 0.0, 7.071068]], columns=['filter_col', 'bool_col', 'float_col'], index=Index([True], name='groupby_col'))
    tm.assert_frame_equal(result, expected)