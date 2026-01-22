import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'Int64'])
def test_astype_copies(dtype):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [1, 2, 3]}, dtype=dtype)
    result = df.astype('int64[pyarrow]', copy=True)
    df.iloc[0, 0] = 100
    expected = DataFrame({'a': [1, 2, 3]}, dtype='int64[pyarrow]')
    tm.assert_frame_equal(result, expected)