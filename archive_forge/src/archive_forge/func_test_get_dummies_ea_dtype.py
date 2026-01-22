import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@td.skip_if_no('pyarrow')
def test_get_dummies_ea_dtype(self):
    for dtype, exp_dtype in [('string[pyarrow]', 'boolean'), ('string[pyarrow_numpy]', 'bool'), (CategoricalDtype(Index(['a'], dtype='string[pyarrow]')), 'boolean'), (CategoricalDtype(Index(['a'], dtype='string[pyarrow_numpy]')), 'bool')]:
        df = DataFrame({'name': Series(['a'], dtype=dtype), 'x': 1})
        result = get_dummies(df)
        expected = DataFrame({'x': 1, 'name_a': Series([True], dtype=exp_dtype)})
        tm.assert_frame_equal(result, expected)