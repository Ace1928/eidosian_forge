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
@pytest.mark.parametrize('value', [2, np.nan, None, float('nan')])
def test_constructor_dict_nan_key(self, value):
    cols = [1, value, 3]
    idx = ['a', value]
    values = [[0, 3], [1, 4], [2, 5]]
    data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
    result = DataFrame(data).sort_values(1).sort_values('a', axis=1)
    expected = DataFrame(np.arange(6, dtype='int64').reshape(2, 3), index=idx, columns=cols)
    tm.assert_frame_equal(result, expected)
    result = DataFrame(data, index=idx).sort_values('a', axis=1)
    tm.assert_frame_equal(result, expected)
    result = DataFrame(data, index=idx, columns=cols)
    tm.assert_frame_equal(result, expected)