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
def test_maybe_convert_objects_timedelta64_nat(self):
    obj = np.timedelta64('NaT', 'ns')
    arr = np.array([obj], dtype=object)
    assert arr[0] is obj
    result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    expected = np.array([obj], dtype='m8[ns]')
    tm.assert_numpy_array_equal(result, expected)