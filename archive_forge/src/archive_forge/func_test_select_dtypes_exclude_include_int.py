import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
@pytest.mark.parametrize('include', [(np.bool_, 'int'), (np.bool_, 'integer'), ('bool', int)])
def test_select_dtypes_exclude_include_int(self, include):
    df = DataFrame({'a': list('abc'), 'b': list(range(1, 4)), 'c': np.arange(3, 6, dtype='int32'), 'd': np.arange(4.0, 7.0, dtype='float64'), 'e': [True, False, True], 'f': pd.date_range('now', periods=3).values})
    exclude = (np.datetime64,)
    result = df.select_dtypes(include=include, exclude=exclude)
    expected = df[['b', 'c', 'e']]
    tm.assert_frame_equal(result, expected)