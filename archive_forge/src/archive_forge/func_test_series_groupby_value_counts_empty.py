from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('columns', [['A', 'B'], ['A', 'B', 'C']])
def test_series_groupby_value_counts_empty(columns):
    df = DataFrame(columns=columns)
    dfg = df.groupby(columns[:-1])
    result = dfg[columns[-1]].value_counts()
    expected = Series([], dtype=result.dtype, name='count')
    expected.index = MultiIndex.from_arrays([[]] * len(columns), names=columns)
    tm.assert_series_equal(result, expected)