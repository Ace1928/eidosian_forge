from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('op', ['__neg__', '__abs__', '__invert__'])
def test_unary_op_does_not_propagate_mask(data, op):
    data, _ = data
    ser = pd.Series(data)
    if op == '__invert__' and data.dtype.kind == 'f':
        msg = "ufunc 'invert' not supported for the input types"
        with pytest.raises(TypeError, match=msg):
            getattr(ser, op)()
        with pytest.raises(TypeError, match=msg):
            getattr(data, op)()
        with pytest.raises(TypeError, match=msg):
            getattr(data._data, op)()
        return
    result = getattr(ser, op)()
    expected = result.copy(deep=True)
    ser[0] = None
    tm.assert_series_equal(result, expected)