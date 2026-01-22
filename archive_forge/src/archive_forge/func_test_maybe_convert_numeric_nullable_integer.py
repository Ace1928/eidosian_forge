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
@pytest.mark.parametrize('convert_to_masked_nullable, exp', [(True, IntegerArray(np.array([2, 0], dtype='i8'), np.array([False, True]))), (False, np.array([2, np.nan], dtype='float64'))])
def test_maybe_convert_numeric_nullable_integer(self, convert_to_masked_nullable, exp):
    arr = np.array([2, np.nan], dtype=object)
    result = lib.maybe_convert_numeric(arr, set(), convert_to_masked_nullable=convert_to_masked_nullable)
    if convert_to_masked_nullable:
        result = IntegerArray(*result)
        tm.assert_extension_array_equal(result, exp)
    else:
        result = result[0]
        tm.assert_numpy_array_equal(result, exp)