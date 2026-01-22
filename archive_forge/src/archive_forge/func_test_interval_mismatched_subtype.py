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
def test_interval_mismatched_subtype(self):
    first = Interval(0, 1, closed='left')
    second = Interval(Timestamp(0), Timestamp(1), closed='left')
    third = Interval(Timedelta(0), Timedelta(1), closed='left')
    arr = np.array([first, second])
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'
    arr = np.array([second, third])
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'
    arr = np.array([first, third])
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'
    flt_interval = Interval(1.5, 2.5, closed='left')
    arr = np.array([first, flt_interval], dtype=object)
    assert lib.infer_dtype(arr, skipna=False) == 'interval'