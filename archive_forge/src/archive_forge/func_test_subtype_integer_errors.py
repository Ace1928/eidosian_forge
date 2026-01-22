import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
def test_subtype_integer_errors(self):
    index = interval_range(-10.0, 10.0)
    dtype = IntervalDtype('uint64', 'right')
    msg = re.escape('Cannot convert interval[float64, right] to interval[uint64, right]; subtypes are incompatible')
    with pytest.raises(TypeError, match=msg):
        index.astype(dtype)