from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['float', 'float64', 'int64', 'Int64', 'boolean'])
@pytest.mark.parametrize('value', [np.nan, pd.NA])
def test_replace_no_replacement_dtypes(self, dtype, value):
    df = DataFrame(np.eye(2), dtype=dtype)
    result = df.replace(to_replace=[None, -np.inf, np.inf], value=value)
    tm.assert_frame_equal(result, df)