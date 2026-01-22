from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_categorical_unchanged(self):
    df = DataFrame(Series(['a', 'b', 'c'], dtype='category', name='A'))
    ser = Series([0, 1, 2], index=[0, 1, 3], name='B')
    result = pd.concat([df, ser], axis=1)
    expected = DataFrame({'A': Series(['a', 'b', 'c', np.nan], dtype='category'), 'B': Series([0, 1, np.nan, 2], dtype='float')})
    tm.assert_equal(result, expected)