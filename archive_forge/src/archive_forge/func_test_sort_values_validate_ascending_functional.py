import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('ascending', [False, 0, 1, True])
def test_sort_values_validate_ascending_functional(self, ascending):
    df = DataFrame({'D': [23, 7, 21]})
    indexer = df['D'].argsort().values
    if not ascending:
        indexer = indexer[::-1]
    expected = df.loc[df.index[indexer]]
    result = df.sort_values(by='D', ascending=ascending)
    tm.assert_frame_equal(result, expected)