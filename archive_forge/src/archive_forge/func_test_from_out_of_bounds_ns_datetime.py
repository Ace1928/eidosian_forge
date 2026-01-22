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
@pytest.mark.parametrize('cls', [datetime, np.datetime64])
def test_from_out_of_bounds_ns_datetime(self, constructor, cls, request, box, frame_or_series):
    if box is list or (frame_or_series is Series and box is dict):
        mark = pytest.mark.xfail(reason='Timestamp constructor has been updated to cast dt64 to non-nano, but DatetimeArray._from_sequence has not', strict=True)
        request.applymarker(mark)
    scalar = datetime(9999, 1, 1)
    exp_dtype = 'M8[us]'
    if cls is np.datetime64:
        scalar = np.datetime64(scalar, 'D')
        exp_dtype = 'M8[s]'
    result = constructor(scalar)
    item = get1(result)
    dtype = tm.get_dtype(result)
    assert type(item) is Timestamp
    assert item.asm8.dtype == exp_dtype
    assert dtype == exp_dtype