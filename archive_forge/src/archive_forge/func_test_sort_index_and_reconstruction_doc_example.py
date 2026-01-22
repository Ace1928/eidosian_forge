import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_and_reconstruction_doc_example(self):
    df = DataFrame({'value': [1, 2, 3, 4]}, index=MultiIndex(levels=[['a', 'b'], ['bb', 'aa']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]))
    assert df.index._is_lexsorted()
    assert not df.index.is_monotonic_increasing
    expected = DataFrame({'value': [2, 1, 4, 3]}, index=MultiIndex(levels=[['a', 'b'], ['aa', 'bb']], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]))
    result = df.sort_index()
    assert result.index.is_monotonic_increasing
    tm.assert_frame_equal(result, expected)
    result = df.sort_index().copy()
    result.index = result.index._sort_levels_monotonic()
    assert result.index.is_monotonic_increasing
    tm.assert_frame_equal(result, expected)