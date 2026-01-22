import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_delete_preserves_rangeindex_middle(self):
    idx = Index(range(3), name='foo')
    result = idx.delete(1)
    expected = idx[::2]
    tm.assert_index_equal(result, expected, exact=True)
    result = idx.delete(-2)
    tm.assert_index_equal(result, expected, exact=True)