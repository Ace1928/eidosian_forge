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
@pytest.mark.parametrize('how,expected', [('right', DataFrame({'A': [100, 200, 300], 'B1': [60, 70, np.nan], 'B2': [600, 700, 800]})), ('outer', DataFrame({'A': [1, 100, 200, 300], 'B1': [80, 60, 70, np.nan], 'B2': [np.nan, 600, 700, 800]}))])
def test_merge_duplicate_suffix(how, expected):
    left_df = DataFrame({'A': [100, 200, 1], 'B': [60, 70, 80]})
    right_df = DataFrame({'A': [100, 200, 300], 'B': [600, 700, 800]})
    result = merge(left_df, right_df, on='A', how=how, suffixes=('_x', '_x'))
    expected.columns = ['A', 'B_x', 'B_x']
    tm.assert_frame_equal(result, expected)