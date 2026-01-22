import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
def test_subtype_float(self, index):
    dtype = IntervalDtype('float64', 'right')
    msg = 'Cannot convert .* to .*; subtypes are incompatible'
    with pytest.raises(TypeError, match=msg):
        index.astype(dtype)