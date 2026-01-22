from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_arr,labels,expected', [([[1, np.nan, 2], [3, 4, 5]], [1, np.nan, 2], np.array([-1, -1, -1], dtype=np.intp)), ([[1, np.nan, 2], [3, 4, 5]], [(np.nan, 4)], np.array([1], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [(1, np.nan)], np.array([0], dtype=np.intp)), ([[1, 2, 3], [np.nan, 4, 5]], [np.nan, 4, 5], np.array([-1, -1, -1], dtype=np.intp))])
def test_get_indexer_with_missing_value(self, index_arr, labels, expected):
    idx = MultiIndex.from_arrays(index_arr)
    result = idx.get_indexer(labels)
    tm.assert_numpy_array_equal(result, expected)