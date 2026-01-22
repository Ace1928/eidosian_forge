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
@pytest.mark.parametrize('data, index, columns, dtype, expected', [(None, list(range(10)), ['a', 'b'], object, np.object_), (None, None, ['a', 'b'], 'int64', np.dtype('int64')), (None, list(range(10)), ['a', 'b'], int, np.dtype('float64')), ({}, None, ['foo', 'bar'], None, np.object_), ({'b': 1}, list(range(10)), list('abc'), int, np.dtype('float64'))])
def test_constructor_dtype(self, data, index, columns, dtype, expected):
    df = DataFrame(data, index, columns, dtype)
    assert df.values.dtype == expected