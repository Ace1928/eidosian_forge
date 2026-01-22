from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
@pytest.mark.parametrize('dtype', dtypes)
def test_binary_input_dispatch_binop(dtype):
    values = np.array([[-1, -1], [1, 1]], dtype='int64')
    df = pd.DataFrame(values, columns=['A', 'B'], index=['a', 'b']).astype(dtype=dtype)
    result = np.add(df, df)
    expected = pd.DataFrame(np.add(values, values), index=df.index, columns=df.columns).astype(dtype)
    tm.assert_frame_equal(result, expected)