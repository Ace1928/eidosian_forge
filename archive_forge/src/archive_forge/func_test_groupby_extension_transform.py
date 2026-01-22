import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_groupby_extension_transform(self, data_for_grouping):
    valid = data_for_grouping[~data_for_grouping.isna()]
    df = pd.DataFrame({'A': [1, 1, 3, 3, 1], 'B': valid})
    result = df.groupby('B').A.transform(len)
    expected = pd.Series([3, 3, 2, 2, 3], name='A')
    self.assert_series_equal(result, expected)