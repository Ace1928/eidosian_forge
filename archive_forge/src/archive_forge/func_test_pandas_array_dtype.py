import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_pandas_array_dtype(self, data):
    result = pd.array(data, dtype=np.dtype(object))
    expected = pd.arrays.NumpyExtensionArray(np.asarray(data, dtype=object))
    tm.assert_equal(result, expected)