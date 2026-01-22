from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
def test_get_indexer_with_NA_values(self, unique_nulls_fixture, unique_nulls_fixture2):
    if unique_nulls_fixture is unique_nulls_fixture2:
        return
    arr = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
    index = Index(arr, dtype=object)
    result = index.get_indexer(Index([unique_nulls_fixture, unique_nulls_fixture2, 'Unknown'], dtype=object))
    expected = np.array([0, 1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)