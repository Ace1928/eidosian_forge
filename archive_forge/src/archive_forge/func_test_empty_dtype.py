from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype,expected', [(np.float64, DataFrame(columns=['a', 'b'], dtype=np.float64)), ('category', DataFrame({'a': Categorical([]), 'b': Categorical([])})), ({'a': 'category', 'b': 'category'}, DataFrame({'a': Categorical([]), 'b': Categorical([])})), ('datetime64[ns]', DataFrame(columns=['a', 'b'], dtype='datetime64[ns]')), ('timedelta64[ns]', DataFrame({'a': Series([], dtype='timedelta64[ns]'), 'b': Series([], dtype='timedelta64[ns]')})), ({'a': np.int64, 'b': np.int32}, DataFrame({'a': Series([], dtype=np.int64), 'b': Series([], dtype=np.int32)})), ({0: np.int64, 1: np.int32}, DataFrame({'a': Series([], dtype=np.int64), 'b': Series([], dtype=np.int32)})), ({'a': np.int64, 1: np.int32}, DataFrame({'a': Series([], dtype=np.int64), 'b': Series([], dtype=np.int32)}))])
@skip_pyarrow
def test_empty_dtype(all_parsers, dtype, expected):
    parser = all_parsers
    data = 'a,b'
    result = parser.read_csv(StringIO(data), header=0, dtype=dtype)
    tm.assert_frame_equal(result, expected)