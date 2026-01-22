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
@pytest.mark.parametrize('value, expected_dtype', [([2 ** 63], np.uint64), ([np.uint64(2 ** 63)], np.uint64), ([2, -1], np.int64), ([2 ** 63, -1], object), ([np.uint8(1)], np.uint8), ([np.uint16(1)], np.uint16), ([np.uint32(1)], np.uint32), ([np.uint64(1)], np.uint64), ([np.uint8(2), np.uint16(1)], np.uint16), ([np.uint32(2), np.uint16(1)], np.uint32), ([np.uint32(2), -1], object), ([np.uint32(2), 1], np.uint64), ([np.uint32(2), np.int32(1)], object)])
def test_maybe_convert_objects_uint(self, value, expected_dtype):
    arr = np.array(value, dtype=object)
    exp = np.array(value, dtype=expected_dtype)
    tm.assert_numpy_array_equal(lib.maybe_convert_objects(arr), exp)