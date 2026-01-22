from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_for_new_dtypes(self, datetime_frame):
    tsframe = datetime_frame.copy().astype(np.float32)
    tsframe.loc[tsframe.index[:5], 'A'] = np.nan
    tsframe.loc[tsframe.index[-5:], 'A'] = np.nan
    zero_filled = tsframe.replace(np.nan, -100000000.0)
    tm.assert_frame_equal(zero_filled, tsframe.fillna(-100000000.0))
    tm.assert_frame_equal(zero_filled.replace(-100000000.0, np.nan), tsframe)
    tsframe.loc[tsframe.index[:5], 'A'] = np.nan
    tsframe.loc[tsframe.index[-5:], 'A'] = np.nan
    tsframe.loc[tsframe.index[:5], 'B'] = np.nan
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = tsframe.fillna(method='bfill')
        tm.assert_frame_equal(result, tsframe.fillna(method='bfill'))