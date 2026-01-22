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
@pytest.mark.parametrize('values', [np.array([1], dtype=np.uint16), np.array([1], dtype=np.uint32), np.array([1], dtype=np.uint64), [np.uint16(1)], [np.uint32(1)], [np.uint64(1)]])
def test_constructor_numpy_uints(self, values):
    value = values[0]
    result = DataFrame(values)
    assert result[0].dtype == value.dtype
    assert result[0][0] == value