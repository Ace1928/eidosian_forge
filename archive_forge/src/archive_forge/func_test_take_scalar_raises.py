import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_take_scalar_raises(self, arr):
    msg = "'indices' must be an array, not a scalar '2'."
    with pytest.raises(ValueError, match=msg):
        arr.take(2)