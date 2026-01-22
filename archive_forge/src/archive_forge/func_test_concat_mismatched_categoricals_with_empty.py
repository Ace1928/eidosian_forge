import pytest
import pandas.core.dtypes.concat as _concat
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_concat_mismatched_categoricals_with_empty():
    ser1 = Series(['a', 'b', 'c'], dtype='category')
    ser2 = Series([], dtype='category')
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = _concat.concat_compat([ser1._values, ser2._values])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = pd.concat([ser1, ser2])._values
    tm.assert_categorical_equal(result, expected)