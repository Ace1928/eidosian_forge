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
def test_is_datetime_dtypes(self):
    ts = pd.date_range('20130101', periods=3)
    tsa = pd.date_range('20130101', periods=3, tz='US/Eastern')
    msg = 'is_datetime64tz_dtype is deprecated'
    assert is_datetime64_dtype('datetime64')
    assert is_datetime64_dtype('datetime64[ns]')
    assert is_datetime64_dtype(ts)
    assert not is_datetime64_dtype(tsa)
    assert not is_datetime64_ns_dtype('datetime64')
    assert is_datetime64_ns_dtype('datetime64[ns]')
    assert is_datetime64_ns_dtype(ts)
    assert is_datetime64_ns_dtype(tsa)
    assert is_datetime64_any_dtype('datetime64')
    assert is_datetime64_any_dtype('datetime64[ns]')
    assert is_datetime64_any_dtype(ts)
    assert is_datetime64_any_dtype(tsa)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert not is_datetime64tz_dtype('datetime64')
        assert not is_datetime64tz_dtype('datetime64[ns]')
        assert not is_datetime64tz_dtype(ts)
        assert is_datetime64tz_dtype(tsa)