import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer', [[True, False], pd.array([True, False], dtype='boolean'), np.array([True, False], dtype=np.bool_)])
def test_bool_raise_length(indexer):
    arr = np.array([1, 2, 3])
    msg = 'Boolean index has wrong length'
    with pytest.raises(IndexError, match=msg):
        check_array_indexer(arr, indexer)