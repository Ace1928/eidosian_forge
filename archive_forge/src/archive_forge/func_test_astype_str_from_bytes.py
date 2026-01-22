import pytest
from pandas import (
import pandas._testing as tm
def test_astype_str_from_bytes():
    idx = Index(['あ', b'a'], dtype='object')
    result = idx.astype(str)
    expected = Index(['あ', 'a'], dtype='object')
    tm.assert_index_equal(result, expected)
    result = Series(idx).astype(str)
    expected = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected)