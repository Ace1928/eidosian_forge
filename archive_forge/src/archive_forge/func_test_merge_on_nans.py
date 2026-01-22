import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.mark.parametrize('func', [lambda x: x, lambda x: to_datetime(x)], ids=['numeric', 'datetime'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_merge_on_nans(self, func, side):
    msg = f'Merge keys contain null values on {side} side'
    nulls = func([1.0, 5.0, np.nan])
    non_nulls = func([1.0, 5.0, 10.0])
    df_null = pd.DataFrame({'a': nulls, 'left_val': ['a', 'b', 'c']})
    df = pd.DataFrame({'a': non_nulls, 'right_val': [1, 6, 11]})
    with pytest.raises(ValueError, match=msg):
        if side == 'left':
            merge_asof(df_null, df, on='a')
        else:
            merge_asof(df, df_null, on='a')