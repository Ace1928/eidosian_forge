import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_change_dtype_series(self, using_infer_string):
    df = pd.DataFrame.from_dict({'Test': ['0.5', True, '0.6']})
    warn = FutureWarning if using_infer_string else None
    with tm.assert_produces_warning(warn, match='Downcasting'):
        df['Test'] = df['Test'].replace([True], [np.nan])
    expected = pd.DataFrame.from_dict({'Test': ['0.5', np.nan, '0.6']})
    tm.assert_frame_equal(df, expected)
    df = pd.DataFrame.from_dict({'Test': ['0.5', None, '0.6']})
    df['Test'] = df['Test'].replace([None], [np.nan])
    tm.assert_frame_equal(df, expected)
    df = pd.DataFrame.from_dict({'Test': ['0.5', None, '0.6']})
    df['Test'] = df['Test'].fillna(np.nan)
    tm.assert_frame_equal(df, expected)