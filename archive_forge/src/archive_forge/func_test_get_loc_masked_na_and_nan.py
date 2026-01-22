import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_loc_masked_na_and_nan(self):
    idx = Index(FloatingArray(np.array([1, 2, 1, np.nan]), mask=np.array([False, False, True, False])))
    result = idx.get_loc(NA)
    assert result == 2
    result = idx.get_loc(np.nan)
    assert result == 3
    idx = Index(FloatingArray(np.array([1, 2, 1.0]), mask=np.array([False, False, True])))
    result = idx.get_loc(NA)
    assert result == 2
    with pytest.raises(KeyError, match='nan'):
        idx.get_loc(np.nan)
    idx = Index(FloatingArray(np.array([1, 2, np.nan]), mask=np.array([False, False, False])))
    result = idx.get_loc(np.nan)
    assert result == 2
    with pytest.raises(KeyError, match='NA'):
        idx.get_loc(NA)