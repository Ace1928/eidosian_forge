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
def test_merge_right_index_right(self):
    left = DataFrame({'a': [1, 2, 3], 'key': [0, 1, 1]})
    right = DataFrame({'b': [1, 2, 3]})
    expected = DataFrame({'a': [1, 2, 3, None], 'key': [0, 1, 1, 2], 'b': [1, 2, 2, 3]}, columns=['a', 'key', 'b'], index=[0, 1, 2, np.nan])
    result = left.merge(right, left_on='key', right_index=True, how='right')
    tm.assert_frame_equal(result, expected)