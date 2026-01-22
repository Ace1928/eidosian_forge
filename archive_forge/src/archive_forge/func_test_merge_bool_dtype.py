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
@pytest.mark.parametrize('how, expected_data', [('inner', [[True, 1, 4], [False, 5, 3]]), ('outer', [[False, 5, 3], [True, 1, 4]]), ('left', [[True, 1, 4], [False, 5, 3]]), ('right', [[False, 5, 3], [True, 1, 4]])])
def test_merge_bool_dtype(self, how, expected_data):
    df1 = DataFrame({'A': [True, False], 'B': [1, 5]})
    df2 = DataFrame({'A': [False, True], 'C': [3, 4]})
    result = merge(df1, df2, how=how)
    expected = DataFrame(expected_data, columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)