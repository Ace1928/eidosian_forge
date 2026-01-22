import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_categorical(frame_or_series):
    exp = frame_or_series(pd.Categorical(['A', 'A', 'B', 'B', np.nan], categories=['A', 'B', 'C']), dtype='category')
    df = frame_or_series(['A', 'A', 'B', 'B', 'C'], dtype='category')
    res = df.where(df != 'C')
    tm.assert_equal(exp, res)