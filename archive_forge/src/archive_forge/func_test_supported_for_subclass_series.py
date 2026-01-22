import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.pytables import (
def test_supported_for_subclass_series(self, tmp_path):
    data = [1, 2, 3]
    sser = tm.SubclassedSeries(data, dtype=np.intp)
    expected = Series(data, dtype=np.intp)
    path = tmp_path / 'temp.h5'
    sser.to_hdf(path, key='ser')
    result = read_hdf(path, 'ser')
    tm.assert_series_equal(result, expected)
    path = tmp_path / 'temp.h5'
    with HDFStore(path) as store:
        store.put('ser', sser)
    result = read_hdf(path, 'ser')
    tm.assert_series_equal(result, expected)