import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_get_loc_overflows(self):
    idx = Index([0, 2, 1])
    val = np.iinfo(np.int64).max + 1
    with pytest.raises(KeyError, match=str(val)):
        idx.get_loc(val)
    with pytest.raises(KeyError, match=str(val)):
        idx._engine.get_loc(val)