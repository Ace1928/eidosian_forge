import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('val', [[1, 2, 3], np.array([1, 2]), (1, 2, 3)])
def test_set_fill_invalid_non_scalar(self, val):
    arr = SparseArray([True, False, True], fill_value=False, dtype=np.bool_)
    msg = 'fill_value must be a scalar'
    with pytest.raises(ValueError, match=msg):
        arr.fill_value = val