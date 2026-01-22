import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.floating import (
def test_repr_array_long():
    data = pd.array([1.0, 2.0, None] * 1000)
    expected = '<FloatingArray>\n[ 1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,\n ...\n <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>,  1.0,  2.0, <NA>]\nLength: 3000, dtype: Float64'
    result = repr(data)
    assert result == expected