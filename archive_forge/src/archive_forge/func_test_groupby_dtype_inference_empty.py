from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_dtype_inference_empty():
    df = DataFrame({'x': [], 'range': np.arange(0, dtype='int64')})
    assert df['x'].dtype == np.float64
    result = df.groupby('x').first()
    exp_index = Index([], name='x', dtype=np.float64)
    expected = DataFrame({'range': Series([], index=exp_index, dtype='int64')})
    tm.assert_frame_equal(result, expected, by_blocks=True)