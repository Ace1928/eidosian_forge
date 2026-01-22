import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_constructor_from_string():
    result = NumpyEADtype.construct_from_string('int64')
    expected = NumpyEADtype(np.dtype('int64'))
    assert result == expected