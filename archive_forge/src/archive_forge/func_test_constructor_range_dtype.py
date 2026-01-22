import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.arm_slow
@pytest.mark.parametrize('dtype', [None, 'uint8', 'category'])
def test_constructor_range_dtype(self, dtype):
    expected = DataFrame({'A': [0, 1, 2, 3, 4]}, dtype=dtype or 'int64')
    result = DataFrame(range(5), columns=['A'], dtype=dtype)
    tm.assert_frame_equal(result, expected)
    result = DataFrame({'A': range(5)}, dtype=dtype)
    tm.assert_frame_equal(result, expected)