import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_missing_category():
    ser = Series([1, 2, 3, 1], dtype='category')
    msg = 'Cannot setitem on a Categorical with a new category \\(-1\\)'
    with pytest.raises(TypeError, match=msg):
        ser.reindex([1, 2, 3, 4, 5], fill_value=-1)