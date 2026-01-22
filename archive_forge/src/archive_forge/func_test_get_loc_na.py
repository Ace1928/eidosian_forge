import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_loc_na(self):
    idx = Index([np.nan, 1, 2], dtype=np.float64)
    assert idx.get_loc(1) == 1
    assert idx.get_loc(np.nan) == 0
    idx = Index([np.nan, 1, np.nan], dtype=np.float64)
    assert idx.get_loc(1) == 1
    msg = "'Cannot get left slice bound for non-unique label: nan'"
    with pytest.raises(KeyError, match=msg):
        idx.slice_locs(np.nan)
    idx = Index([np.nan, 1, np.nan, np.nan], dtype=np.float64)
    assert idx.get_loc(1) == 1
    msg = "'Cannot get left slice bound for non-unique label: nan"
    with pytest.raises(KeyError, match=msg):
        idx.slice_locs(np.nan)