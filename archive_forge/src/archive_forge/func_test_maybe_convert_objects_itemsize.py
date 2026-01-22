import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('data0', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
@pytest.mark.parametrize('data1', [True, 1, 1.0, 1.0 + 1j, np.int8(1), np.int16(1), np.int32(1), np.int64(1), np.float16(1), np.float32(1), np.float64(1), np.complex64(1), np.complex128(1)])
def test_maybe_convert_objects_itemsize(self, data0, data1):
    data = [data0, data1]
    arr = np.array(data, dtype='object')
    common_kind = np.result_type(type(data0), type(data1)).kind
    kind0 = 'python' if not hasattr(data0, 'dtype') else data0.dtype.kind
    kind1 = 'python' if not hasattr(data1, 'dtype') else data1.dtype.kind
    if kind0 != 'python' and kind1 != 'python':
        kind = common_kind
        itemsize = max(data0.dtype.itemsize, data1.dtype.itemsize)
    elif is_bool(data0) or is_bool(data1):
        kind = 'bool' if is_bool(data0) and is_bool(data1) else 'object'
        itemsize = ''
    elif is_complex(data0) or is_complex(data1):
        kind = common_kind
        itemsize = 16
    else:
        kind = common_kind
        itemsize = 8
    expected = np.array(data, dtype=f'{kind}{itemsize}')
    result = lib.maybe_convert_objects(arr)
    tm.assert_numpy_array_equal(result, expected)