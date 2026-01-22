import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_insert_edges_preserves_rangeindex(self):
    idx = Index(range(4, 9, 2))
    result = idx.insert(0, 2)
    expected = Index(range(2, 9, 2))
    tm.assert_index_equal(result, expected, exact=True)
    result = idx.insert(3, 10)
    expected = Index(range(4, 11, 2))
    tm.assert_index_equal(result, expected, exact=True)