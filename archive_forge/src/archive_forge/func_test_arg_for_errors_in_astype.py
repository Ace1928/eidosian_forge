from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_arg_for_errors_in_astype(self):
    ser = Series([1, 2, 3])
    msg = "Expected value of kwarg 'errors' to be one of \\['raise', 'ignore'\\]\\. Supplied value is 'False'"
    with pytest.raises(ValueError, match=msg):
        ser.astype(np.float64, errors=False)
    ser.astype(np.int8, errors='raise')