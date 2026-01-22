import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_constructor_with_data(any_numpy_array):
    nparr = any_numpy_array
    arr = NumpyExtensionArray(nparr)
    assert arr.dtype.numpy_dtype == nparr.dtype