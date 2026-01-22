import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values', [[], list(range(5))])
def test_drop_duplicates_no_duplicates(any_numpy_dtype, keep, values):
    tc = Series(values, dtype=np.dtype(any_numpy_dtype))
    expected = Series([False] * len(tc), dtype='bool')
    if tc.dtype == 'bool':
        tc = tc[:2]
        expected = expected[:2]
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    result_dropped = tc.drop_duplicates(keep=keep)
    tm.assert_series_equal(result_dropped, tc)
    assert result_dropped is not tc