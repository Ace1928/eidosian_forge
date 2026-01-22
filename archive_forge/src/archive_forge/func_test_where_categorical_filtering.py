from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_categorical_filtering(self):
    df = DataFrame(data=[[0, 0], [1, 1]], columns=['a', 'b'])
    df['b'] = df['b'].astype('category')
    result = df.where(df['a'] > 0)
    expected = df.copy().astype({'a': 'float'})
    expected.loc[0, :] = np.nan
    tm.assert_equal(result, expected)