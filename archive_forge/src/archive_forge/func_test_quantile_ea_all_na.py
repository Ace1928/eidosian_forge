import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_ea_all_na(self, request, obj, index):
    obj.iloc[:] = index._na_value
    assert np.all(obj.dtypes == index.dtype)
    indexer = np.arange(len(index), dtype=np.intp)
    np.random.default_rng(2).shuffle(indexer)
    obj = obj.iloc[indexer]
    qs = [0.5, 0, 1]
    result = self.compute_quantile(obj, qs)
    expected = index.take([-1, -1, -1], allow_fill=True, fill_value=index._na_value)
    expected = Series(expected, index=qs, name='A')
    expected = type(obj)(expected)
    tm.assert_equal(result, expected)