import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_drop_nan_with_multi_index():
    df = pd.DataFrame([[np.nan, 0, 1]], columns=['a', 'b', 'c'])
    df = df.set_index(['a', 'b'])
    result = df.groupby(['a', 'b'], dropna=False).first()
    expected = df
    tm.assert_frame_equal(result, expected)