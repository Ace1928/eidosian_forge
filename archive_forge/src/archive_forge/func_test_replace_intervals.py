from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_intervals(self, using_infer_string):
    df = DataFrame({'a': [pd.Interval(0, 1), pd.Interval(0, 1)]})
    warning = FutureWarning if using_infer_string else None
    with tm.assert_produces_warning(warning, match='Downcasting'):
        result = df.replace({'a': {pd.Interval(0, 1): 'x'}})
    expected = DataFrame({'a': ['x', 'x']})
    tm.assert_frame_equal(result, expected)