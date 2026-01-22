import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_index_string_inference(self):
    pytest.importorskip('pyarrow')
    dtype = 'string[pyarrow_numpy]'
    expected = Index(['a', 'b'], dtype=dtype)
    with pd.option_context('future.infer_string', True):
        ser = Index(['a', 'b'])
    tm.assert_index_equal(ser, expected)
    expected = Index(['a', 1], dtype='object')
    with pd.option_context('future.infer_string', True):
        ser = Index(['a', 1])
    tm.assert_index_equal(ser, expected)