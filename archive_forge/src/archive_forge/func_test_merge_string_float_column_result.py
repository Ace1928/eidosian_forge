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
def test_merge_string_float_column_result():
    df1 = DataFrame([[1, 2], [3, 4]], columns=Index(['a', 114.0]))
    df2 = DataFrame([[9, 10], [11, 12]], columns=['x', 'y'])
    result = merge(df2, df1, how='inner', left_index=True, right_index=True)
    expected = DataFrame([[9, 10, 1, 2], [11, 12, 3, 4]], columns=Index(['x', 'y', 'a', 114.0]))
    tm.assert_frame_equal(result, expected)