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
def test_multi_func(df):
    col1 = df['A']
    col2 = df['B']
    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean(numeric_only=True)
    expected = df.groupby(['A', 'B']).mean()
    tm.assert_frame_equal(agged.loc[:, ['C', 'D']], expected.loc[:, ['C', 'D']], check_names=False)
    df = DataFrame({'v1': np.random.default_rng(2).standard_normal(6), 'v2': np.random.default_rng(2).standard_normal(6), 'k1': np.array(['b', 'b', 'b', 'a', 'a', 'a']), 'k2': np.array(['1', '1', '1', '2', '2', '2'])}, index=['one', 'two', 'three', 'four', 'five', 'six'])
    grouped = df.groupby(['k1', 'k2'])
    grouped.agg('sum')