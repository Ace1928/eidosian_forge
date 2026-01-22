import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('data, data_exp', [(['a', 'b', 'c'], ['b', 'b', 'c']), (['a'], ['b'])])
def test_replace_categorical_inplace(self, data, data_exp):
    result = pd.Series(data, dtype='category')
    msg = 'with CategoricalDtype is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result.replace(to_replace='a', value='b', inplace=True)
    expected = pd.Series(data_exp, dtype='category')
    tm.assert_series_equal(result, expected)