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
@pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
def test_other_datetime_unit(self, unit):
    df1 = DataFrame({'entity_id': [101, 102]})
    ser = Series([None, None], index=[101, 102], name='days')
    dtype = f'datetime64[{unit}]'
    if unit in ['D', 'h', 'm']:
        exp_dtype = 'datetime64[s]'
    else:
        exp_dtype = dtype
    df2 = ser.astype(exp_dtype).to_frame('days')
    assert df2['days'].dtype == exp_dtype
    result = df1.merge(df2, left_on='entity_id', right_index=True)
    days = np.array(['nat', 'nat'], dtype=exp_dtype)
    days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
    exp = DataFrame({'entity_id': [101, 102], 'days': days}, columns=['entity_id', 'days'])
    assert exp['days'].dtype == exp_dtype
    tm.assert_frame_equal(result, exp)