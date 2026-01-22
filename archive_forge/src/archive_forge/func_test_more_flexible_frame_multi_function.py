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
def test_more_flexible_frame_multi_function(df):
    grouped = df.groupby('A')
    exmean = grouped.agg({'C': 'mean', 'D': 'mean'})
    exstd = grouped.agg({'C': 'std', 'D': 'std'})
    expected = concat([exmean, exstd], keys=['mean', 'std'], axis=1)
    expected = expected.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)
    d = {'C': ['mean', 'std'], 'D': ['mean', 'std']}
    result = grouped.aggregate(d)
    tm.assert_frame_equal(result, expected)
    result = grouped.aggregate({'C': 'mean', 'D': ['mean', 'std']})
    expected = grouped.aggregate({'C': 'mean', 'D': ['mean', 'std']})
    tm.assert_frame_equal(result, expected)

    def numpymean(x):
        return np.mean(x)

    def numpystd(x):
        return np.std(x, ddof=1)
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        d = {'C': 'mean', 'D': {'foo': 'mean', 'bar': 'std'}}
        grouped.aggregate(d)
    d = {'C': ['mean'], 'D': [numpymean, numpystd]}
    grouped.aggregate(d)