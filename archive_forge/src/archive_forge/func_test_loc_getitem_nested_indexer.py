import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('indexer_type_1', (list, tuple, set, slice, np.ndarray, Series, Index))
@pytest.mark.parametrize('indexer_type_2', (list, tuple, set, slice, np.ndarray, Series, Index))
def test_loc_getitem_nested_indexer(self, indexer_type_1, indexer_type_2):

    def convert_nested_indexer(indexer_type, keys):
        if indexer_type == np.ndarray:
            return np.array(keys)
        if indexer_type == slice:
            return slice(*keys)
        return indexer_type(keys)
    a = [10, 20, 30]
    b = [1, 2, 3]
    index = MultiIndex.from_product([a, b])
    df = DataFrame(np.arange(len(index), dtype='int64'), index=index, columns=['Data'])
    keys = ([10, 20], [2, 3])
    types = (indexer_type_1, indexer_type_2)
    indexer = tuple((convert_nested_indexer(indexer_type, k) for indexer_type, k in zip(types, keys)))
    if indexer_type_1 is set or indexer_type_2 is set:
        with pytest.raises(TypeError, match='as an indexer is not supported'):
            df.loc[indexer, 'Data']
        return
    else:
        result = df.loc[indexer, 'Data']
    expected = Series([1, 2, 4, 5], name='Data', index=MultiIndex.from_product(keys))
    tm.assert_series_equal(result, expected)