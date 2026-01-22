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
@pytest.mark.parametrize('arr', [np.array([np.timedelta64('nat'), np.datetime64('2011-01-02')], dtype=object), np.array([np.datetime64('2011-01-02'), np.timedelta64('nat')], dtype=object), np.array([np.datetime64('2011-01-01'), Timestamp('2011-01-02')]), np.array([Timestamp('2011-01-02'), np.datetime64('2011-01-01')]), np.array([np.nan, Timestamp('2011-01-02'), 1.1]), np.array([np.nan, '2011-01-01', Timestamp('2011-01-02')], dtype=object), np.array([np.datetime64('nat'), np.timedelta64(1, 'D')], dtype=object), np.array([np.timedelta64(1, 'D'), np.datetime64('nat')], dtype=object)])
def test_infer_datetimelike_dtype_mixed(self, arr):
    assert lib.infer_dtype(arr, skipna=False) == 'mixed'