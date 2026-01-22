import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('numeric_only', [True, False])
def test_corrwith_mixed_dtypes(self, numeric_only):
    df = DataFrame({'a': [1, 4, 3, 2], 'b': [4, 6, 7, 3], 'c': ['a', 'b', 'c', 'd']})
    s = Series([0, 6, 7, 3])
    if numeric_only:
        result = df.corrwith(s, numeric_only=numeric_only)
        corrs = [df['a'].corr(s), df['b'].corr(s)]
        expected = Series(data=corrs, index=['a', 'b'])
        tm.assert_series_equal(result, expected)
    else:
        with pytest.raises(ValueError, match='could not convert string to float'):
            df.corrwith(s, numeric_only=numeric_only)