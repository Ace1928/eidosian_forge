from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_numeric_should_not_fallback_to_positional(self, any_numeric_dtype):
    dtype = any_numeric_dtype
    idx = Index([1, 0, 1], dtype=dtype)
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=idx)
    result = df[1]
    expected = DataFrame([[1, 3], [4, 6]], columns=Index([1, 1], dtype=dtype))
    tm.assert_frame_equal(result, expected, check_exact=True)