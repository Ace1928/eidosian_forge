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
def test_merge_ea_int_and_float_numpy():
    df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
    df2 = DataFrame([1.5])
    expected = DataFrame(columns=[0], dtype='Int64')
    with tm.assert_produces_warning(UserWarning, match='You are merging'):
        result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(UserWarning, match='You are merging'):
        result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype('float64'))
    df2 = DataFrame([1.0])
    expected = DataFrame([1], columns=[0], dtype='Int64')
    result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)
    result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype('float64'))