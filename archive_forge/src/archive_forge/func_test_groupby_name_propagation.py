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
def test_groupby_name_propagation(df):

    def summarize(df, name=None):
        return Series({'count': 1, 'mean': 2, 'omissions': 3}, name=name)

    def summarize_random_name(df):
        return Series({'count': 1, 'mean': 2, 'omissions': 3}, name=df.iloc[0]['A'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        metrics = df.groupby('A').apply(summarize)
    assert metrics.columns.name is None
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        metrics = df.groupby('A').apply(summarize, 'metrics')
    assert metrics.columns.name == 'metrics'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        metrics = df.groupby('A').apply(summarize_random_name)
    assert metrics.columns.name is None