from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_no_name_column_conflict():
    df = DataFrame({'name': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2], 'name2': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1], 'value': range(9, -1, -1)})
    grouped = df.groupby(['name', 'name2'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grouped.apply(lambda x: x.sort_values('value', inplace=True))