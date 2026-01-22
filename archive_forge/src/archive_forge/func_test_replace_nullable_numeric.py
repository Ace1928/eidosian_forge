import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_nullable_numeric(self):
    floats = pd.Series([1.0, 2.0, 3.999, 4.4], dtype=pd.Float64Dtype())
    assert floats.replace({1.0: 9}).dtype == floats.dtype
    assert floats.replace(1.0, 9).dtype == floats.dtype
    assert floats.replace({1.0: 9.0}).dtype == floats.dtype
    assert floats.replace(1.0, 9.0).dtype == floats.dtype
    res = floats.replace(to_replace=[1.0, 2.0], value=[9.0, 10.0])
    assert res.dtype == floats.dtype
    ints = pd.Series([1, 2, 3, 4], dtype=pd.Int64Dtype())
    assert ints.replace({1: 9}).dtype == ints.dtype
    assert ints.replace(1, 9).dtype == ints.dtype
    assert ints.replace({1: 9.0}).dtype == ints.dtype
    assert ints.replace(1, 9.0).dtype == ints.dtype
    with pytest.raises(TypeError, match='Invalid value'):
        ints.replace({1: 9.5})
    with pytest.raises(TypeError, match='Invalid value'):
        ints.replace(1, 9.5)