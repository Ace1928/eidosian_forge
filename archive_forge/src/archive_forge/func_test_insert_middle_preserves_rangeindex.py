import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_insert_middle_preserves_rangeindex(self):
    idx = Index(range(0, 3, 2))
    result = idx.insert(1, 1)
    expected = Index(range(3))
    tm.assert_index_equal(result, expected, exact=True)
    idx = idx * 2
    result = idx.insert(1, 2)
    expected = expected * 2
    tm.assert_index_equal(result, expected, exact=True)