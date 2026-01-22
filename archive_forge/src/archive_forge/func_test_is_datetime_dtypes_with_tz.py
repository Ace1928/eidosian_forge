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
@pytest.mark.parametrize('tz', ['US/Eastern', 'UTC'])
def test_is_datetime_dtypes_with_tz(self, tz):
    dtype = f'datetime64[ns, {tz}]'
    assert not is_datetime64_dtype(dtype)
    msg = 'is_datetime64tz_dtype is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        assert is_datetime64tz_dtype(dtype)
    assert is_datetime64_ns_dtype(dtype)
    assert is_datetime64_any_dtype(dtype)