import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_integer
from pandas.core.arrays import IntegerArray
from pandas.core.arrays.integer import (
def test_to_integer_array_dtype_keyword(constructor):
    result = constructor([1, 2], dtype='Int8')
    assert result.dtype == Int8Dtype()
    result = constructor(np.array([1, 2], dtype='int8'), dtype='Int32')
    assert result.dtype == Int32Dtype()