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
@pytest.mark.parametrize('how, sort, expected', [('inner', False, DataFrame({'a': [20, 10], 'b': [200, 100]}, index=[2, 1])), ('inner', True, DataFrame({'a': [10, 20], 'b': [100, 200]}, index=[1, 2])), ('left', False, DataFrame({'a': [20, 10, 0], 'b': [200, 100, np.nan]}, index=[2, 1, 0])), ('left', True, DataFrame({'a': [0, 10, 20], 'b': [np.nan, 100, 200]}, index=[0, 1, 2])), ('right', False, DataFrame({'a': [np.nan, 10, 20], 'b': [300, 100, 200]}, index=[3, 1, 2])), ('right', True, DataFrame({'a': [10, 20, np.nan], 'b': [100, 200, 300]}, index=[1, 2, 3])), ('outer', False, DataFrame({'a': [0, 10, 20, np.nan], 'b': [np.nan, 100, 200, 300]}, index=[0, 1, 2, 3])), ('outer', True, DataFrame({'a': [0, 10, 20, np.nan], 'b': [np.nan, 100, 200, 300]}, index=[0, 1, 2, 3]))])
def test_merge_on_indexes(self, left_df, right_df, how, sort, expected):
    result = merge(left_df, right_df, left_index=True, right_index=True, how=how, sort=sort)
    tm.assert_frame_equal(result, expected)