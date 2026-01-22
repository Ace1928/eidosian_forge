import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_engineless_lookup(self):
    idx = RangeIndex(2, 10, 3)
    assert idx.get_loc(5) == 1
    tm.assert_numpy_array_equal(idx.get_indexer([2, 8]), ensure_platform_int(np.array([0, 2])))
    with pytest.raises(KeyError, match='3'):
        idx.get_loc(3)
    assert '_engine' not in idx._cache
    with pytest.raises(KeyError, match="'a'"):
        idx.get_loc('a')
    assert '_engine' not in idx._cache