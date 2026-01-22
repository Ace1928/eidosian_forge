import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_astype_nan_raises(self):
    arr = SparseArray([1.0, np.nan])
    with pytest.raises(ValueError, match='Cannot convert non-finite'):
        arr.astype(int)