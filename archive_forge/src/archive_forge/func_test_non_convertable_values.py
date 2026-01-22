from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_non_convertable_values(self):
    msg = "Could not convert string 'foo' to numeric"
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric('foo')
    msg = 'argument must be a string or a number'
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric({})
    with pytest.raises(TypeError, match=msg):
        nanops._ensure_numeric([])