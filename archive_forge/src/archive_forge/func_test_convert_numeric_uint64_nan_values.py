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
@pytest.mark.parametrize('convert_to_masked_nullable', [True, False])
def test_convert_numeric_uint64_nan_values(self, coerce, convert_to_masked_nullable):
    arr = np.array([2 ** 63, 2 ** 63 + 1], dtype=object)
    na_values = {2 ** 63}
    expected = np.array([np.nan, 2 ** 63 + 1], dtype=float) if coerce else arr.copy()
    result = lib.maybe_convert_numeric(arr, na_values, coerce_numeric=coerce, convert_to_masked_nullable=convert_to_masked_nullable)
    if convert_to_masked_nullable and coerce:
        expected = IntegerArray(np.array([0, 2 ** 63 + 1], dtype='u8'), np.array([True, False], dtype='bool'))
        result = IntegerArray(*result)
    else:
        result = result[0]
    tm.assert_almost_equal(result, expected)