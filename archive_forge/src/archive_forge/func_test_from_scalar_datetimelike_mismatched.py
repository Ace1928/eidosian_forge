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
@pytest.mark.parametrize('cls', [np.datetime64, np.timedelta64])
def test_from_scalar_datetimelike_mismatched(self, constructor, cls):
    scalar = cls('NaT', 'ns')
    dtype = {np.datetime64: 'm8[ns]', np.timedelta64: 'M8[ns]'}[cls]
    if cls is np.datetime64:
        msg1 = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
    else:
        msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
    msg = '|'.join(['Cannot cast', msg1])
    with pytest.raises(TypeError, match=msg):
        constructor(scalar, dtype=dtype)
    scalar = cls(4, 'ns')
    with pytest.raises(TypeError, match=msg):
        constructor(scalar, dtype=dtype)