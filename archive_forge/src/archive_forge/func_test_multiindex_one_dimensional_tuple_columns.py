import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer', [('a',), 'a'])
def test_multiindex_one_dimensional_tuple_columns(self, indexer):
    mi = MultiIndex.from_tuples([('a', 'A'), ('b', 'A')])
    obj = DataFrame([1, 2], index=mi)
    obj.loc[indexer, :] = 0
    expected = DataFrame([0, 2], index=mi)
    tm.assert_frame_equal(obj, expected)