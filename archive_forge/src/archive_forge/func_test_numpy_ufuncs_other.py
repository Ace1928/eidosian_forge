import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
from pandas.core.arrays import BooleanArray
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
@pytest.mark.parametrize('func', [np.isfinite, np.isinf, np.isnan, np.signbit], ids=lambda x: x.__name__)
def test_numpy_ufuncs_other(index, func):
    if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
        if func in (np.isfinite, np.isinf, np.isnan):
            result = func(index)
            assert isinstance(result, np.ndarray)
            out = np.empty(index.shape, dtype=bool)
            func(index, out=out)
            tm.assert_numpy_array_equal(out, result)
        else:
            with tm.external_error_raised(TypeError):
                func(index)
    elif isinstance(index, PeriodIndex):
        with tm.external_error_raised(TypeError):
            func(index)
    elif is_numeric_dtype(index) and (not (is_complex_dtype(index) and func is np.signbit)):
        result = func(index)
        if not isinstance(index.dtype, np.dtype):
            assert isinstance(result, BooleanArray)
        else:
            assert isinstance(result, np.ndarray)
        out = np.empty(index.shape, dtype=bool)
        func(index, out=out)
        if not isinstance(index.dtype, np.dtype):
            tm.assert_numpy_array_equal(out, result._data)
        else:
            tm.assert_numpy_array_equal(out, result)
    elif len(index) == 0:
        pass
    else:
        with tm.external_error_raised(TypeError):
            func(index)