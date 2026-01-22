import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_floating_array_disallows_float16():
    arr = np.array([1, 2], dtype=np.float16)
    mask = np.array([False, False])
    msg = 'FloatingArray does not support np.float16 dtype'
    with pytest.raises(TypeError, match=msg):
        FloatingArray(arr, mask)