import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [object, pytest.param('string[pyarrow_numpy]', marks=td.skip_if_no('pyarrow')), pytest.param('string[pyarrow]', marks=td.skip_if_no('pyarrow'))])
def test_size_strings(dtype):
    df = DataFrame({'a': ['a', 'a', 'b'], 'b': 'a'}, dtype=dtype)
    result = df.groupby('a')['b'].size()
    exp_dtype = 'Int64' if dtype == 'string[pyarrow]' else 'int64'
    expected = Series([2, 1], index=Index(['a', 'b'], name='a', dtype=dtype), name='b', dtype=exp_dtype)
    tm.assert_series_equal(result, expected)