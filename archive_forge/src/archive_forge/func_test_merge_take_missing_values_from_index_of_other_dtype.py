from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_take_missing_values_from_index_of_other_dtype(self):
    left = DataFrame({'a': [1, 2, 3], 'key': Categorical(['a', 'a', 'b'], categories=list('abc'))})
    right = DataFrame({'b': [1, 2, 3]}, index=CategoricalIndex(['a', 'b', 'c']))
    result = left.merge(right, left_on='key', right_index=True, how='right')
    expected = DataFrame({'a': [1, 2, 3, None], 'key': Categorical(['a', 'a', 'b', 'c']), 'b': [1, 1, 2, 3]}, index=[0, 1, 2, np.nan])
    expected = expected.reindex(columns=['a', 'key', 'b'])
    tm.assert_frame_equal(result, expected)