from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('ufunc', [np.positive, np.negative, np.abs])
def test_unary_ufunc_dunder_equivalence(self, data, ufunc):
    attr = {np.positive: '__pos__', np.negative: '__neg__', np.abs: '__abs__'}[ufunc]
    exc = None
    try:
        result = getattr(data, attr)()
    except Exception as err:
        exc = err
        with pytest.raises((type(exc), TypeError)):
            ufunc(data)
    else:
        alt = ufunc(data)
        tm.assert_extension_array_equal(result, alt)