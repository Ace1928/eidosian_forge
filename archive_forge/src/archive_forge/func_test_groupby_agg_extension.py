import numpy as np
import pytest
from pandas.core.dtypes.common import is_bool_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.boolean import BooleanDtype
from pandas.tests.extension import base
def test_groupby_agg_extension(self, data_for_grouping):
    df = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': data_for_grouping})
    expected = df.iloc[[0, 2, 4]]
    expected = expected.set_index('A')
    result = df.groupby('A').agg({'B': 'first'})
    self.assert_frame_equal(result, expected)
    result = df.groupby('A').agg('first')
    self.assert_frame_equal(result, expected)
    result = df.groupby('A').first()
    self.assert_frame_equal(result, expected)