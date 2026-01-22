import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_dtype_idempotent(any_numpy_dtype):
    dtype = NumpyEADtype(any_numpy_dtype)
    result = NumpyEADtype(dtype)
    assert result == dtype