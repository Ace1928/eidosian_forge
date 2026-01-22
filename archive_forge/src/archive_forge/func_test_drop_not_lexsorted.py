import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_not_lexsorted():
    tuples = [('a', ''), ('b1', 'c1'), ('b2', 'c2')]
    lexsorted_mi = MultiIndex.from_tuples(tuples, names=['b', 'c'])
    assert lexsorted_mi._is_lexsorted()
    df = pd.DataFrame(columns=['a', 'b', 'c', 'd'], data=[[1, 'b1', 'c1', 3], [1, 'b2', 'c2', 4]])
    df = df.pivot_table(index='a', columns=['b', 'c'], values='d')
    df = df.reset_index()
    not_lexsorted_mi = df.columns
    assert not not_lexsorted_mi._is_lexsorted()
    tm.assert_index_equal(lexsorted_mi, not_lexsorted_mi)
    with tm.assert_produces_warning(PerformanceWarning):
        tm.assert_index_equal(lexsorted_mi.drop('a'), not_lexsorted_mi.drop('a'))