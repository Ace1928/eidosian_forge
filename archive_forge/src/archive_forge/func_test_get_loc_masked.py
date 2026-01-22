import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('val, val2', [(4, 5), (4, 4), (4, NA), (NA, NA)])
def test_get_loc_masked(self, val, val2, any_numeric_ea_and_arrow_dtype):
    idx = Index([1, 2, 3, val, val2], dtype=any_numeric_ea_and_arrow_dtype)
    result = idx.get_loc(2)
    assert result == 1
    with pytest.raises(KeyError, match='9'):
        idx.get_loc(9)