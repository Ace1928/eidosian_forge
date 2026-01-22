import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_from_obscure_array(dtype, array_likes):
    arr, data = array_likes
    cls = {'M8[ns]': DatetimeArray, 'm8[ns]': TimedeltaArray}[dtype]
    depr_msg = f'{cls.__name__}.__init__ is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = cls(arr)
    result = cls._from_sequence(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    if not isinstance(data, memoryview):
        func = {'M8[ns]': pd.to_datetime, 'm8[ns]': pd.to_timedelta}[dtype]
        result = func(arr).array
        expected = func(data).array
        tm.assert_equal(result, expected)
    idx_cls = {'M8[ns]': DatetimeIndex, 'm8[ns]': TimedeltaIndex}[dtype]
    result = idx_cls(arr)
    expected = idx_cls(data)
    tm.assert_index_equal(result, expected)