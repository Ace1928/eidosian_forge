import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_frame_describe_multikey(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])
    result = grouped.describe()
    desc_groups = []
    for col in tsframe:
        group = grouped[col].describe()
        group_col = MultiIndex(levels=[[col], group.columns], codes=[[0] * len(group.columns), range(len(group.columns))])
        group = DataFrame(group.values, columns=group_col, index=group.index)
        desc_groups.append(group)
    expected = pd.concat(desc_groups, axis=1)
    tm.assert_frame_equal(result, expected)
    groupedT = tsframe.groupby({'A': 0, 'B': 0, 'C': 1, 'D': 1}, axis=1)
    result = groupedT.describe()
    expected = tsframe.describe().T
    expected.index = MultiIndex(levels=[[0, 1], expected.index], codes=[[0, 0, 1, 1], range(len(expected.index))])
    tm.assert_frame_equal(result, expected)