import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.indexers import check_array_indexer
@pytest.mark.parametrize('indexer', [None, Ellipsis, slice(0, 3), (None,)])
def test_pass_through_non_array_likes(indexer):
    arr = np.array([1, 2, 3])
    result = check_array_indexer(arr, indexer)
    assert result == indexer