import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_base(self):
    idx = CategoricalIndex(list('cab'), categories=list('cab'))
    expected = np.arange(len(idx), dtype=np.intp)
    actual = idx.get_indexer(idx)
    tm.assert_numpy_array_equal(expected, actual)
    with pytest.raises(ValueError, match='Invalid fill method'):
        idx.get_indexer(idx, method='invalid')