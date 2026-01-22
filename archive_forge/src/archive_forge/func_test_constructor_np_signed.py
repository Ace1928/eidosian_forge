import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_np_signed(self, any_signed_int_numpy_dtype):
    scalar = np.dtype(any_signed_int_numpy_dtype).type(1)
    result = Index([scalar])
    expected = Index([1], dtype=any_signed_int_numpy_dtype)
    tm.assert_index_equal(result, expected, exact=True)