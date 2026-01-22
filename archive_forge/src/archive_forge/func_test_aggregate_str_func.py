import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('groupbyfunc', [lambda x: x.weekday(), [lambda x: x.month, lambda x: x.weekday()]])
def test_aggregate_str_func(tsframe, groupbyfunc):
    grouped = tsframe.groupby(groupbyfunc)
    result = grouped['A'].agg('std')
    expected = grouped['A'].std()
    tm.assert_series_equal(result, expected)
    result = grouped.aggregate('var')
    expected = grouped.var()
    tm.assert_frame_equal(result, expected)
    result = grouped.agg({'A': 'var', 'B': 'std', 'C': 'mean', 'D': 'sem'})
    expected = DataFrame({'A': grouped['A'].var(), 'B': grouped['B'].std(), 'C': grouped['C'].mean(), 'D': grouped['D'].sem()})
    tm.assert_frame_equal(result, expected)