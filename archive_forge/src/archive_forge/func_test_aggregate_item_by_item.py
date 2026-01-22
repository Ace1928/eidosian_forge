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
def test_aggregate_item_by_item(df):
    grouped = df.groupby('A')
    aggfun_0 = lambda ser: ser.size
    result = grouped.agg(aggfun_0)
    foosum = (df.A == 'foo').sum()
    barsum = (df.A == 'bar').sum()
    K = len(result.columns)
    exp = Series(np.array([foosum] * K), index=list('BCD'), name='foo')
    tm.assert_series_equal(result.xs('foo'), exp)
    exp = Series(np.array([barsum] * K), index=list('BCD'), name='bar')
    tm.assert_almost_equal(result.xs('bar'), exp)

    def aggfun_1(ser):
        return ser.size
    result = DataFrame().groupby(df.A).agg(aggfun_1)
    assert isinstance(result, DataFrame)
    assert len(result) == 0