from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_group_apply_once_per_group2(capsys):
    expected = 2
    df = DataFrame({'group_by_column': [0, 0, 0, 0, 1, 1, 1, 1], 'test_column': ['0', '2', '4', '6', '8', '10', '12', '14']}, index=['0', '2', '4', '6', '8', '10', '12', '14'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        df.groupby('group_by_column', group_keys=False).apply(lambda df: print('function_called'))
    result = capsys.readouterr().out.count('function_called')
    assert result == expected