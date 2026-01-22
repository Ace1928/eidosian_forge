from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_float_to_uint_negatives_raise(self, float_numpy_dtype, any_unsigned_int_numpy_dtype):
    arr = np.arange(5).astype(float_numpy_dtype) - 3
    ser = Series(arr)
    msg = 'Cannot losslessly cast from .* to .*'
    with pytest.raises(ValueError, match=msg):
        ser.astype(any_unsigned_int_numpy_dtype)
    with pytest.raises(ValueError, match=msg):
        ser.to_frame().astype(any_unsigned_int_numpy_dtype)
    with pytest.raises(ValueError, match=msg):
        Index(ser).astype(any_unsigned_int_numpy_dtype)
    with pytest.raises(ValueError, match=msg):
        ser.array.astype(any_unsigned_int_numpy_dtype)