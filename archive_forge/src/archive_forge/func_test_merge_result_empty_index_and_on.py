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
def test_merge_result_empty_index_and_on():
    df1 = DataFrame({'a': [1], 'b': [2]}).set_index(['a', 'b'])
    df2 = DataFrame({'b': [1]}).set_index(['b'])
    expected = DataFrame({'a': [], 'b': []}, dtype=np.int64).set_index(['a', 'b'])
    result = merge(df1, df2, left_on=['b'], right_index=True)
    tm.assert_frame_equal(result, expected)
    result = merge(df2, df1, left_index=True, right_on=['b'])
    tm.assert_frame_equal(result, expected)