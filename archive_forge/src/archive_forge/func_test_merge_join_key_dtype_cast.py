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
def test_merge_join_key_dtype_cast(self):
    df1 = DataFrame({'key': [1], 'v1': [10]})
    df2 = DataFrame({'key': [2], 'v1': [20]})
    df = merge(df1, df2, how='outer')
    assert df['key'].dtype == 'int64'
    df1 = DataFrame({'key': [True], 'v1': [1]})
    df2 = DataFrame({'key': [False], 'v1': [0]})
    df = merge(df1, df2, how='outer')
    assert df['key'].dtype == 'bool'
    df1 = DataFrame({'val': [1]})
    df2 = DataFrame({'val': [2]})
    lkey = np.array([1])
    rkey = np.array([2])
    df = merge(df1, df2, left_on=lkey, right_on=rkey, how='outer')
    assert df['key_0'].dtype == np.dtype(int)