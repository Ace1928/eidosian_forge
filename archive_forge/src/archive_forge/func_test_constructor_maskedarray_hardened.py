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
def test_constructor_maskedarray_hardened(self):
    mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
    result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
    expected = DataFrame({'A': [np.nan, np.nan], 'B': [np.nan, np.nan]}, columns=['A', 'B'], index=[1, 2], dtype=float)
    tm.assert_frame_equal(result, expected)
    mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
    result = DataFrame(mat_hard, columns=['A', 'B'], index=[1, 2])
    expected = DataFrame({'A': [1.0, 1.0], 'B': [1.0, 1.0]}, columns=['A', 'B'], index=[1, 2], dtype=float)
    tm.assert_frame_equal(result, expected)