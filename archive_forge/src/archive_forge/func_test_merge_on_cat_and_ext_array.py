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
def test_merge_on_cat_and_ext_array():
    right = DataFrame({'a': Series([pd.Interval(0, 1), pd.Interval(1, 2)], dtype='interval')})
    left = right.copy()
    left['a'] = left['a'].astype('category')
    result = merge(left, right, how='inner', on='a')
    expected = right.copy()
    tm.assert_frame_equal(result, expected)